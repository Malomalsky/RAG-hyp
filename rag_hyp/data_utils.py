"""Dataset preparation utilities."""

import os
import json
import pickle
import logging
from typing import List, Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import faiss

from .hyperbolic_utils import (
    DEFAULT_DTYPE,
    poincare_ball,
    get_heuristic_hyperbolic_embedding,
    calculate_hyperbolic_centroid,
)
from . import viz_utils

def load_raw_dataset(file_path):
    """Загружает и парсит JSONL датасет."""
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try: data_list.append(json.loads(line.strip()))
                except json.JSONDecodeError as e: print(f"[Line {line_number}] JSON decoding error: {e}")
    except FileNotFoundError:
         print(f"Ошибка: Raw dataset файл не найден: {file_path}")
         return None
    print(f"Загружено {len(data_list)} записей из {file_path}")
    return data_list

def clean_data(dataset):
    """Очищает данные, извлекает diffs, msgs, ids."""
    diffs, msgs, diff_ids = [], [], []
    for entry in dataset:
        diff, msg, diff_id = entry.get('diff'), entry.get('msg'), entry.get('diff_id')
        if diff and msg and diff_id is not None: 
            try:
                diff_ids.append(str(diff_id)) # Сохраняем ID как строку (хэш)
                diffs.append(str(diff).replace('<nl>', '\n').strip())
                msgs.append(' '.join(str(msg).replace('<nl>', '\n').split()))
            except (ValueError, TypeError):
                 pass 
    print(f"Осталось {len(diffs)} валидных записей после очистки.")
    return diffs, msgs, diff_ids

def split_dataset(diffs, messages, diff_ids, file_paths_list, config: dict):
    """Разделяет датасет на train/validation/test сеты, включая file_paths."""
    train_size = config.get("train_split_ratio", 0.7)
    val_size = config.get("validation_split_ratio", 0.15)
    test_size = config.get("test_split_ratio", 0.15)
    random_state=config["random_state"]
    
    expected_total = len(diffs)
    if expected_total == 0: print("Ошибка: Нет данных для разделения."); return None
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        print(f"Warning: Сумма размеров ({train_size}+{val_size}+{test_size}) != 1. Проверьте *_split_ratio в CONFIG.")
        # Можно добавить нормализацию или вернуть ошибку
        # total = train_size + val_size + test_size
        # train_size /= total; val_size /= total; test_size /= total
        return None # Лучше остановить, если размеры неправильные
        
    train_val_diffs, test_diffs, \
    train_val_msgs, test_msgs, \
    train_val_ids, test_ids, \
    train_val_paths, test_paths = train_test_split(
        test_size=test_size, random_state=random_state)
        
    if len(train_val_diffs) > 0:
        train_val_sum = train_size + val_size
        relative_train_size = train_size / train_val_sum if train_val_sum > 0 else 0.0
        if relative_train_size >= 1.0: relative_train_size = 0.999999 
        if relative_train_size > 0:
            train_diffs, val_diffs, \
            train_msgs, val_msgs, \
            train_ids, val_ids, \
            train_paths, val_paths = train_test_split(
                train_size=relative_train_size, random_state=random_state)
        else: 
             train_diffs, train_msgs, train_ids, train_paths = [], [], [], [] 
             val_diffs, val_msgs, val_ids, val_paths = train_val_diffs, train_val_msgs, train_val_ids, train_val_paths 
    else:
        train_diffs, val_diffs, train_msgs, val_msgs, train_ids, val_ids, train_paths, val_paths = [], [], [], [], [], [], [], [] 
        
    print(f"Разделение данных: Train={len(train_diffs)}, Validation={len(val_diffs)}, Test={len(test_diffs)}")
    return {
        'train': {'diffs': train_diffs, 'messages': train_msgs, 'diff_ids': train_ids, 'file_paths': train_paths},
        'validation': {'diffs': val_diffs, 'messages': val_msgs, 'diff_ids': val_ids, 'file_paths': val_paths},
        'test': {'diffs': test_diffs, 'messages': test_msgs, 'diff_ids': test_ids, 'file_paths': test_paths},
    }

def serialize_splits(dataset_splits: dict, config: dict):
    """Сериализует сплиты датасета в файлы, включая file_paths."""
    output_dir = config["prepared_data_dir"]
    os.makedirs(output_dir, exist_ok=True)
    success = True
    for split_name, data in dataset_splits.items():
        try:
            file_path = config[f"{split_name}_split_path"]
        except KeyError:
            print(f"Ошибка: Ключ '{split_name}_split_path' не найден в CONFIG.")
            success = False
            continue
        try:
            if 'file_paths' not in data:
                print(f"Warning: Ключ 'file_paths' отсутствует в данных для сплита '{split_name}'. Сохранение без него.")
            with open(file_path, 'wb') as f: pickle.dump(data, f)
            print(f"Сериализован {split_name} сет в {file_path}")
        except Exception as e: 
            print(f"Ошибка при сериализации {split_name} сета: {e}")
            success = False
    return success
    
def load_embedding_model(model_name: str, device: str) -> tuple:
    """Загружает модель и токенизатор для эмбеддингов."""
    try:
        print(f"Загрузка модели эмбеддингов: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print(f"Модель эмбеддингов загружена на {device}.")
        return tokenizer, model
    except Exception as e:
        print(f"Критическая ошибка загрузки модели эмбеддингов {model_name}: {e}")
        return None, None

def generate_semantic_embeddings(diffs: List[str], tokenizer: AutoTokenizer, model: AutoModel, output_path: str, batch_size: int, device: str, config: dict) -> Optional[np.ndarray]:
    """Генерирует и сохраняет семантические эмбеддинги с оптимизацией памяти."""
    if os.path.exists(output_path):
        print(f"Семантические эмбеддинги уже существуют: {output_path}. Загрузка...")
        try:
            embeddings = np.load(output_path)
            if len(embeddings) == len(diffs):
                print(f"Загружены эмбеддинги размера: {embeddings.shape}. Размер совпадает.")
                return embeddings
            else:
                print(f"Warning: Размер загруженных эмб. ({len(embeddings)}) НЕ совпадает с кол-вом диффов ({len(diffs)}). Пересоздаем...")
        except Exception as e:
            print(f"Ошибка загрузки эмбеддингов {output_path}: {e}. Будет выполнена перегенерация.")
            
    print("Генерация семантических эмбеддингов (с оптимизацией памяти)...")
    
    total_embeddings = len(diffs)
    if total_embeddings == 0:
        print("Нет диффов для генерации эмбеддингов.")
        return np.empty((0, 0), dtype=np.float32) # Возвращаем пустой массив
        
    question_encoder_max_length = config.get("question_encoder_max_length", 512) 
    
    embeddings_np = None # Инициализируем None
    embedding_dim = -1
    # all_embeddings = [] # Убираем список для накопления
    
    num_batches = (len(diffs) + batch_size - 1) // batch_size
    model.eval() # Убедимся, что модель в режиме eval
    try:
        for i in tqdm(range(num_batches), desc="Генерация семантических эмбеддингов"):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, total_embeddings)
            batch_diffs = diffs[start_index:end_index]
            
            encoded_input = tokenizer(batch_diffs, padding=True, truncation=True, return_tensors='pt', max_length=question_encoder_max_length)
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            batch_embeddings = model_output.pooler_output if hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None else model_output.last_hidden_state[:, 0, :]
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            # Определяем размерность и создаем массив при первом батче
            if embeddings_np is None:
                embedding_dim = batch_embeddings_np.shape[1]
                embeddings_np = np.empty((total_embeddings, embedding_dim), dtype=np.float32)
                print(f"Определена размерность эмбеддинга: {embedding_dim}. Создан массив {embeddings_np.shape}.")
            
            # Записываем текущий батч в предвыделенный массив
            embeddings_np[start_index:end_index] = batch_embeddings_np
            
        
        # Проверяем, был ли массив создан (на случай пустого входа)
        if embeddings_np is None:
             print("Предупреждение: Не удалось создать массив эмбеддингов (возможно, пустой входной список diffs).")
             return np.empty((0, embedding_dim if embedding_dim > 0 else 0), dtype=np.float32)
             
        print(f"Сгенерированы эмбеддинги размера: {embeddings_np.shape}")
        # Сохранение
        try:
            np.save(output_path, embeddings_np)
            print(f"Семантические эмбеддинги сохранены в {output_path}")
            return embeddings_np
        except Exception as e:
            print(f"Ошибка сохранения семантических эмбеддингов: {e}")
            return None
    except Exception as e:
        print(f"Ошибка во время генерации эмбеддингов: {e}")
        import traceback
        traceback.print_exc() # Выводим полный traceback для отладки
        return None

def build_faiss_index(embeddings: np.ndarray, index_path: str):
    """Строит и сохраняет FAISS индекс."""
    if embeddings is None or embeddings.size == 0:
        print("Ошибка: Нет эмбеддингов для построения FAISS индекса.")
        return False # Возвращаем False при ошибке
    if os.path.exists(index_path):
        print(f"FAISS индекс уже существует: {index_path}. Пропуск построения.")
        return True # Возвращаем True, если существует
        
    print("Построение FAISS индекса...")
    try:
        # Нормализация важна для IndexFlatIP
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        
        index = faiss.IndexFlatIP(embeddings_normalized.shape[1])
        index.add(embeddings_normalized)
        print(f"FAISS индекс построен с {index.ntotal} векторами.")
        # Сохранение
        faiss.write_index(index, index_path)
        print(f"FAISS индекс сохранен в {index_path}")
        return True
    except Exception as e:
        print(f"Ошибка построения/сохранения FAISS индекса: {e}")
        return False

def prepare_structural_embeddings_for_all(file_paths_list: List[List[str]], 
                                              output_path: str,
                                              alpha: float, 
                                              angle_hash_modulo: int,
                                              config: dict):
    """Вычисляет и сохраняет гиперболические центроиды для всех данных."""
    
    embedding_dim = config.get("hyperbolic_embedding_dim", 2)
    dtype = DEFAULT_DTYPE # Используем глобальный float32 для train.py

    # Проверка существующего файла 
    if os.path.exists(output_path):
        logging.info(f"Структурные эмбеддинги (центроиды) уже существуют: {output_path}. Загрузка...")
        try:
            existing_embeddings = torch.load(output_path)
            # Проверяем не только количество, но и размерность
            if len(existing_embeddings) == len(file_paths_list) and existing_embeddings.shape[1] == embedding_dim:
                 logging.info(f"Размер загруженных эмбеддингов ({existing_embeddings.shape}) совпадает. Пропуск создания.")
                 return True 
            else:
                 logging.warning(f"Размер/формат загруженных эмб. ({existing_embeddings.shape}) НЕ совпадает ({len(file_paths_list)}, {embedding_dim}). Пересоздаем...")
        except Exception as e:
             logging.error(f"Ошибка при загрузке/проверке существующих эмбеддингов: {e}. Пересоздаем...")

    logging.info(f"Начало подготовки структурных эмбеддингов (центроидов)...")
    num_embeddings = len(file_paths_list)
    if num_embeddings == 0:
        logging.error("Нет записей для генерации структурных эмбеддингов.")
        return False

    # Создаем тензор нужного размера и типа
    all_centroids = torch.empty((num_embeddings, embedding_dim), dtype=dtype)
    processed_count = 0

    logging.info(f"Вычисление {num_embeddings} гиперболических центроидов...")
    centroid_max_iter = config.get("centroid_max_iterations", 50)
    centroid_lr = config.get("centroid_lr", 0.5)
    centroid_eps = config.get("centroid_convergence_eps", 1e-4)
    for i, paths_for_entry in tqdm(enumerate(file_paths_list), total=num_embeddings, desc="Генерация структурных центроидов"):
        try:
            # 1. Вычисляем эмбеддинги для всех путей в текущей записи
            individual_embeddings = []
            if paths_for_entry: # Только если список путей не пуст
                for path_str in paths_for_entry:
                    # Используем ту же эвристику для индивидуальных точек
                    h_point = get_heuristic_hyperbolic_embedding(
                        path_str, alpha=alpha, angle_hash_modulo=angle_hash_modulo, 
                        embedding_dim=embedding_dim, dtype=dtype, manifold=poincare_ball
                    )
                    individual_embeddings.append(h_point)
            
            # 2. Вычисляем центроид для этих точек
            h_centroid = calculate_hyperbolic_centroid(
                individual_embeddings, 
                embedding_dim=embedding_dim, 
                max_iterations=centroid_max_iter,
                lr=centroid_lr,
            ) 
            
            # 3. Записываем центроид
            all_centroids[i] = h_centroid
            processed_count += 1
        except Exception as e:
            logging.error(f"Ошибка вычисления центроида для индекса {i}: {e}. Используется дефолтный (0,0).", exc_info=True)
            all_centroids[i] = torch.zeros(embedding_dim, dtype=dtype) # Записываем ноль при ошибке

    logging.info(f"Тензор центроидов создан: {all_centroids.shape}. Обработано: {processed_count}/{num_embeddings}.")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logging.info(f"Структурные эмбеддинги (центроиды) сохранены в {output_path}")
        return True
    except Exception as e:
        logging.error(f"Ошибка сохранения тензора структурных центроидов: {e}", exc_info=True)
        return False

def prepare_all_data(config: dict) -> bool:
    """Оркестрирует весь процесс подготовки данных, читая предобработанный JSONL."""
    prep_dir = config["prepared_data_dir"]
    os.makedirs(prep_dir, exist_ok=True)

    jsonl_path = config["raw_dataset_path"] # Теперь это путь к JSONL
    # struct_emb_path = config["structural_embeddings_output_path"] # Путь понадобится позже

    # 1. Проверка наличия основного JSONL
    if not os.path.exists(jsonl_path):
        logging.error(f"Основной файл данных JSONL не найден: {jsonl_path}. Создайте его с помощью prepare_dataset.py или укажите правильный путь.")
        return False
    # if not os.path.exists(struct_emb_path):
    #     logging.error(f"Файл структурных эмбеддингов не найден: {struct_emb_path}. Запустите prepare_dataset.py")
    #     return False

    # 2. Чтение данных из JSONL
    logging.info(f"Чтение данных из {jsonl_path}...")
    diffs, msgs, diff_ids, file_paths_list = [], [], [], []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Чтение JSONL"):
                try:
                    record = json.loads(line)
                    # Проверяем наличие всех необходимых полей
                    if all(k in record for k in ["diff_id", "msg", "diff", "file_paths"]):
                        diff_ids.append(record["diff_id"]) # Строка
                        msgs.append(record["msg"])       # Строка
                        diffs.append(record["diff"])      # Строка
                        file_paths_list.append(record["file_paths"]) # Список строк
                    else:
                        logging.warning(f"Пропущена запись из-за отсутствия полей в JSONL: {record.get('diff_id')}")
                except json.JSONDecodeError as e:
                    logging.error(f"Ошибка парсинга JSON строки при чтении train.py: {e}. Строка: {line.strip()}")
    except Exception as e:
        logging.error(f"Ошибка при чтении файла {jsonl_path} в train.py: {e}")
        return False

    if not diffs:
        logging.error("Не найдено валидных записей в JSONL файле.")
        return False
    logging.info(f"Прочитано {len(diffs)} валидных записей из JSONL.")

    max_entries_to_use = config.get("max_raw_entries", None) 
    if max_entries_to_use is not None and len(diffs) > max_entries_to_use:
        logging.info(f"Используем только первые {max_entries_to_use} записей из {len(diffs)} для обучения/обработки.")
        diffs = diffs[:max_entries_to_use]
        msgs = msgs[:max_entries_to_use]
        diff_ids = diff_ids[:max_entries_to_use]
        file_paths_list = file_paths_list[:max_entries_to_use]
        logging.info(f"Размер датасета после ограничения: {len(diffs)} записей.")

    # 3. Разделение на сплиты и сохранение (если нужно)
    split_paths_needed = [config["train_split_path"], config["validation_split_path"], config["test_split_path"]]
    if not all(os.path.exists(p) for p in split_paths_needed):
        logging.info("Разделение данных на train/validation/test...")
        dataset_splits = split_dataset(diffs, msgs, diff_ids, file_paths_list, config)
        if not dataset_splits or not serialize_splits(dataset_splits, config):
            logging.error("Ошибка: Не удалось разделить/сохранить сплиты.")
            return False
    else:
        logging.info("Файлы с разделенными данными уже существуют.")

    # 4. Генерация семантических эмбеддингов (только если отсутствуют)
    sem_emb_path = config["semantic_embeddings_path"]
    if not os.path.exists(sem_emb_path):
        logging.info("Генерация семантических эмбеддингов...")
        sem_tokenizer, sem_model = load_embedding_model(config["semantic_embedding_model"], config["device"])
        if sem_tokenizer is None: return False
        semantic_embeddings = generate_semantic_embeddings(
             diffs, sem_tokenizer, sem_model, sem_emb_path,
             batch_size=config.get("embedding_batch_size", 16),
             device=config["device"],
             config=config
         )
        if semantic_embeddings is None: return False
        del sem_tokenizer, sem_model # Освобождаем память
        import gc; gc.collect()
    else:
        logging.info(f"Семантические эмбеддинги уже существуют: {sem_emb_path}. Проверка размера...")
        try:
            semantic_embeddings = np.load(sem_emb_path)
            if len(semantic_embeddings) != len(diffs):
                 logging.error(f"Размер сем. эмб. ({len(semantic_embeddings)}) не совпадает с кол-вом записей в JSONL ({len(diffs)}). Удалите эмбеддинги и перезапустите.")
                 return False
            logging.info(f"Размер семантических эмбеддингов совпадает ({len(semantic_embeddings)}).")
        except Exception as e:
            logging.error(f"Ошибка загрузки семантических эмбеддингов: {e}")
            return False

    # 5. Построение FAISS индекса (только если отсутствует)
    faiss_path = config["faiss_index_path"]
    if not os.path.exists(faiss_path):
        logging.info("Построение FAISS индекса...")
        # Загружаем эмбеддинги, если их еще нет в памяти
        if 'semantic_embeddings' not in locals():
            try:
                semantic_embeddings = np.load(sem_emb_path)
            except Exception as e:
                logging.error(f"Ошибка загрузки сем. эмб. для FAISS: {e}")
                return False
        if not build_faiss_index(semantic_embeddings, faiss_path):
            return False
    else:
        logging.info(f"FAISS индекс уже существует: {faiss_path}.")

    # 6. Создание Knowledge Base Dataset (только если отсутствует)
    kb_path = config["knowledge_base_dataset_path"]
    if not (os.path.exists(kb_path) and os.path.exists(os.path.join(kb_path, "dataset_info.json"))):
        logging.info(f"Создание KB Dataset в {kb_path}...")
        titles = [f"diff_{did}" for did in diff_ids] # Используем исходные ID
        # Загружаем эмбеддинги, если их еще нет в памяти
        if 'semantic_embeddings' not in locals():
            try:
                semantic_embeddings = np.load(sem_emb_path)
            except Exception as e:
                logging.error(f"Ошибка загрузки сем. эмб. для KB Dataset: {e}")
                return False
        try:
            # Убедимся, что число эмбеддингов совпадает с числом текстов
            if len(semantic_embeddings) != len(diffs):
                 raise ValueError(f"Число сем. эмб. ({len(semantic_embeddings)}) не совпадает с числом diffs ({len(diffs)}) для KB Dataset.")
            kb_dataset = Dataset.from_dict({'title': titles, 'text': diffs, 'embeddings': [e for e in semantic_embeddings]})
            kb_dataset.save_to_disk(kb_path)
            logging.info(f"KB Dataset сохранен в {kb_path}.")
            del kb_dataset # Освобождаем память
        except Exception as e:
            logging.error(f"Ошибка создания/сохранения KB Dataset: {e}")
            import traceback; traceback.print_exc();
            return False
    else:
        logging.info(f"KB Dataset уже существует: {kb_path}.")

    # 7. Генерация/Проверка структурных эмбеддингов
    struct_emb_path = config["structural_embeddings_output_path"]
    if not os.path.exists(struct_emb_path):
        logging.info(f"Генерация структурных эмбеддингов в {struct_emb_path}...")
        struct_ready = prepare_structural_embeddings_for_all(
            file_paths_list=file_paths_list, 
            output_path=struct_emb_path,
            alpha=config["alpha_depth_scaling"],
            angle_hash_modulo=config["angle_hash_modulo"],
            config=config 
        )
        if not struct_ready:
            logging.error("Не удалось создать структурные эмбеддинги.")
            return False
    else:
        # Если файл существует, проверяем размер
        logging.info(f"Проверка существующих структурных эмбеддингов: {struct_emb_path}")
        try:
            struct_embeddings = torch.load(struct_emb_path)
            if len(struct_embeddings) != len(diffs):
                 logging.error(f"Размер структ. эмб. ({len(struct_embeddings)}) не совпадает с кол-вом записей в JSONL ({len(diffs)}). Удалите файл {struct_emb_path} и перезапустите.")
                 return False
            logging.info(f"Структурные эмбеддинги найдены и размер совпадает ({len(struct_embeddings)}).")
            
            # Отрисовка диска Пуанкаре с центроидами, если модуль визуализации доступен
            if _viz_utils_available:
                try:
                    logging.info("Отрисовка диска Пуанкаре с центроидами...")
                    # Создаем директорию для фигур, если она не существует
                    figures_dir = os.path.join(config["prepared_data_dir"], "figures")
                    os.makedirs(figures_dir, exist_ok=True)
                    
                    # Выбираем подмножество центроидов для отрисовки (не более 1000 для производительности)
                    sample_size = min(1000, len(struct_embeddings))
                    indices = np.random.choice(len(struct_embeddings), size=sample_size, replace=False)
                    
                    sampled_centroids = struct_embeddings[indices]
                    sampled_file_paths = [file_paths_list[i] for i in indices]
                    
                    # Отрисовываем диск Пуанкаре
                    poincare_disk_path = os.path.join(figures_dir, "poincare_disk.png")
                    viz_utils.plot_poincare_disk(
                        centroids=sampled_centroids, 
                        file_paths=sampled_file_paths,
                        output_path=poincare_disk_path,
                        figsize=(8, 8),
                        dpi=300
                    )
                    logging.info(f"Диск Пуанкаре отрисован и сохранен в {poincare_disk_path}")
                except Exception as e:
                    logging.warning(f"Ошибка при отрисовке диска Пуанкаре: {e}")
            
            del struct_embeddings
        except Exception as e:
            logging.error(f"Ошибка загрузки/проверки структурных эмбеддингов: {e}")
            return False

    return True


def load_split_data(config: dict, split_name: str) -> Optional[Dict]:
    """Загружает данные для конкретного сплита, включая file_paths."""
    try:
        file_path = config[f"{split_name}_split_path"]
    except KeyError:
        logging.error(f"Ключ '{split_name}_split_path' не найден в CONFIG.") # Используем logging
        return None
    try:
        with open(file_path, 'rb') as f: split_data = pickle.load(f)
        logging.info(f"Данные для '{split_name}' загружены из {file_path}") # Используем logging
        
        num_entries = len(split_data.get('messages', [])) # Определяем количество записей
        
       
        if 'diffs' in split_data:
            split_data['question_diffs'] = split_data['diffs'] 
        else:
            logging.warning(f"Ключ 'diffs' не найден в {file_path}")
            split_data['question_diffs'] = [""] * num_entries
            
        
        if 'file_paths' in split_data:
            split_data['question_file_paths'] = split_data['file_paths']
        else:
            # Если file_paths отсутствуют, создаем список пустых списков
            logging.warning(f"Ключ 'file_paths' не найден в {file_path}. Будут использованы пустые списки.")
            split_data['question_file_paths'] = [[] for _ in range(num_entries)]
            
        return split_data
    except FileNotFoundError:
        logging.error(f"Файл с данными сплита '{split_name}' не найден: {file_path}")
        return None
    except Exception as e: 
        logging.error(f"Ошибка загрузки '{split_name}' data из {file_path}: {e}") # Используем logging
        return None

def tokenize_dataset_for_training(dataset_dict: Dict, tokenizer: RagTokenizer, config: dict) -> Optional[Dataset]:
    """Токенизирует данные и создает Dataset, сохраняя question_diffs и question_file_paths."""
    try:
        # Проверяем наличие необходимых ключей
        required_keys = ['diffs', 'messages', 'question_diffs', 'question_file_paths']
        if not all(key in dataset_dict for key in required_keys):
             logging.error("В словаре данных отсутствуют необходимые ключи ('diffs', 'messages', 'question_diffs', 'question_file_paths').")
             return None
             
        dataset = Dataset.from_dict({
            'input_text': dataset_dict['diffs'],
            'target_text': dataset_dict['messages'],
            'question_diffs': dataset_dict['question_diffs'], 
        })
        
        question_encoder_max_length = config.get("question_encoder_max_length", 512)
        generator_labels_max_length = config.get("generator_labels_max_length", 128)
        
        def tokenize_fn(examples):
            # Токенизация входа для question_encoder
            model_inputs = tokenizer.question_encoder(examples['input_text'], max_length=question_encoder_max_length, padding="max_length", truncation=True)
            # Токенизация выхода для generator (labels)
            with tokenizer.generator.as_target_tokenizer():
                labels = tokenizer.generator(examples['target_text'], max_length=generator_labels_max_length, padding="max_length", truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            # Передаем question_file_paths и question_diffs
            model_inputs["question_file_paths"] = examples["question_file_paths"]
            model_inputs["question_diffs"] = examples["question_diffs"]
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["input_text", "target_text"]
        )
        # Указываем формат torch для основных колонок
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        # Колонки question_diffs и question_file_paths остаются в формате Python list
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Ошибка при токенизации датасета: {e}", exc_info=True) 
        # import traceback; traceback.print_exc()
        return None
    
