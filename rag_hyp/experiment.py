# -*- coding: utf-8 -*-
"""
Основной скрипт для подготовки данных, инициализации и обучения
модели RAG с гиперболическим переранжированием.
"""

import os
import json
import pickle
import math
import hashlib
import re
import time
from typing import List, Dict, Tuple, Optional, Set, Union, Any, Callable

import torch
import geoopt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import logging
import evaluate
import nltk
from functools import lru_cache
import torch.nn.functional as F
from scipy.stats import spearmanr
import functools
import glob

# Импортируем модуль визуализации
try:
    from . import viz_utils
    _viz_utils_available = True
except Exception:  # pragma: no cover - optional dependency
    _viz_utils_available = False
    print("Модуль viz_utils не найден. Визуализация будет отключена.")

try:
    import faiss
    _faiss_available = True
except ImportError:
    _faiss_available = False

try:
    import datasets
    from datasets import Dataset, load_from_disk
    _datasets_available = True
except ImportError:
    _datasets_available = False

if not _faiss_available or not _datasets_available:
    raise ImportError(
        "Для работы скрипта необходимы библиотеки faiss и datasets.\n" 
        "Установите их: pip install datasets faiss-cpu (или faiss-gpu)"
    )

from transformers import (
    RagConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration, 
    DPRQuestionEncoder, BartForConditionalGeneration,
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, 
    BatchEncoding
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.rag.modeling_rag import RetrievAugLMMarginOutput
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, GenerateOutput

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
CONFIG = {
    # Модели
    "rag_model_name": "facebook/rag-sequence-base", 
    "semantic_embedding_model": "microsoft/codebert-base", # Для генерации семантических эмб.
    
    # Данные
    "prepared_data_dir": "prepared_data",
    "raw_dataset_path": "processed_python_data.jsonl", # Используем большой датасет
    "max_raw_entries": 1000, # Лимит записей для обработки (None для без лимита)
    "cleaned_diffs_path": "prepared_data/cleaned_diffs.pkl",
    "cleaned_msgs_path": "prepared_data/cleaned_msgs.pkl",
    "diff_ids_path": "combined_data/diff_ids.pkl",
    "semantic_embeddings_path": "prepared_data/kb_semantic_embeddings.npy", 
    "faiss_index_path": "prepared_data/kb_index.faiss", 
    "knowledge_base_dataset_path": "prepared_data/kb_dataset", 
    "structural_embeddings_output_path": "prepared_data/kb_structural_embeddings_py.pt",
    "train_split_path": "prepared_data/train_data.pkl",
    "validation_split_path": "prepared_data/validation_data.pkl",
    "test_split_path": "prepared_data/test_data.pkl",
    "train_split_ratio": 0.8, # Добавлено
    "validation_split_ratio": 0.10, # Добавлено
    "test_split_ratio": 0.10, # Добавлено
    "curriculum_weight": 0.05,
    # Параметры Токенизации
    "question_encoder_max_length": 512,
    "generator_labels_max_length": 128,

    # Параметры Ретривера/Переранжирования
    "k_to_rerank": 20,
    "n_final": 5, # Должно совпадать с n_docs в модели RAG
    "rerank_weight": 0.1, # Вес структурного скора (0.0 для baseline)
    
    # Параметры Гиперболического Эмбеддинга
    "alpha_depth_scaling": 0.1,  # Устаревший параметр для tanh
    "angle_hash_modulo": 10000,
    "hyperbolic_embedding_dim": 10,
    "centroid_max_iterations": 50,  
    "centroid_lr": 0.2,             
    "centroid_convergence_eps": 1e-5,
    "centroid_clip_threshold": 0.999,
    "depth_beta": 0.5,             
    "depth_gamma": 1.0,            
    "softmax_temperature": 0.07,   
    "spearman_weight": 0.1,        
   
    
    # Обучение
    "output_dir_base": "training_output", # Добавлено - базовая папка для вывода
    "epochs": 40, # Установите желаемое количество эпох
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5, 
    "warmup_steps": 100,
    "weight_decay": 0.05,
    "early_stopping_patience": 3, 
    "save_strategy": "epoch", 
    "load_best_model_at_end": True, 
    "metric_for_best_model": "eval_rougeL", 
    "greater_is_better": True, 
    "save_total_limit": 2, 
    "logging_steps": 50, # Шаг логирования training loss
    "embedding_batch_size": 16, # Батч для генерации эмбеддингов
    "report_to": "none", # Отключаем внешние трекеры

    # Параметры Генерации (для Оценки/Инференса)
    "generation_max_length": 128, # Макс. длина генерируемого сообщения
    "generation_num_beams": 4, # Количество лучей для beam search
    
    # Другое
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "random_state": 42,

    # Существующие параметры
    'embedding_dim': 512,
    'hyperbolic_centroid_method': 'mean',
    'hyperbolic_curvature': 0.7,
     # Смещение глубины
    "alpha": 0.1,  # Параметр масштабирования для старой формулы radius = tanh(alpha * depth)
    "beta": 0.5,   # Параметр для новой формулы radius = 1 - exp(-beta * (depth + gamma))
    "gamma": 0.2,  # Смещение для глубины в новой формуле
    "temperature": 0.1,  # Температура для мягкого смешивания скоров
  # Вес для регуляризации на основе корреляции Спирмена
}
# ==============================================================================
# 2. ГИПЕРБОЛИЧЕСКИЕ И ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ==============================================================================

DEFAULT_DTYPE = torch.float32 # Используем float32

poincare_ball = geoopt.PoincareBall(c=1.0, learnable=True)
try:
    poincare_ball.k.requires_grad_(True) 
except AttributeError:
    pass # OK если нет setter'а

def project_to_poincare_ball(vectors: torch.Tensor, manifold: geoopt.PoincareBall = poincare_ball, dim: int = -1) -> torch.Tensor:
    """Проецирует евклидовы векторы на диск Пуанкаре."""
    vectors_typed = vectors.to(DEFAULT_DTYPE)
    return manifold.projx(vectors_typed, dim=dim)

def poincare_distance(vectors1: torch.Tensor, vectors2: torch.Tensor, manifold: geoopt.PoincareBall = poincare_ball, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Вычисляет расстояние Пуанкаре."""
    vec1_typed = vectors1.to(DEFAULT_DTYPE)
    vec2_typed = vectors2.to(DEFAULT_DTYPE)
    return manifold.dist(vec1_typed, vec2_typed, dim=dim, keepdim=keepdim)

# Обновляем: принимаем k как тензор/скаляр, чтобы градиент шёл через k.
def poincare_inner_product(x: torch.Tensor, y: torch.Tensor, k):
    """
    Вычисляет гиперболическое скалярное произведение между векторами x и y в диске Пуанкаре.
    
    Гиперболическое скалярное произведение в модели Пуанкаре определяется как:
    <x,y>_p = 4 * (1 - k*|x|^2) * (1 - k*|y|^2) / ((1 - k*|x|^2) + (1 - k*|y|^2))^2 * <x,y>
    
    где <x,y> - обычное евклидово скалярное произведение.
    
    Args:
        x (torch.Tensor): Первый тензор размера (..., dim)
        y (torch.Tensor): Второй тензор размера (..., dim)
        k (float): Кривизна пространства Пуанкаре (по умолчанию 1.0)
        
    Returns:
        torch.Tensor: Гиперболическое скалярное произведение размера (...)
    """
    # Приводим k к тензору той же dtype/устройства, если это скаляр
    if not torch.is_tensor(k):
        k = torch.tensor(float(k), dtype=x.dtype, device=x.device)

    # Проверяем совместимость размерностей
    if x.size(-1) != y.size(-1):
        raise ValueError(f"Последняя размерность тензоров должна совпадать: {x.size(-1)} vs {y.size(-1)}")
    
    # Вычисляем квадрат нормы для каждого вектора
    x_norm_sq = torch.sum(x * x, dim=-1)
    y_norm_sq = torch.sum(y * y, dim=-1)
    
    # Проверяем, что векторы находятся в диске Пуанкаре
    if torch.any(x_norm_sq >= 1.0/k) or torch.any(y_norm_sq >= 1.0/k):
        # Корректируем векторы, если они выходят за пределы диска
        # Это может произойти из-за численной нестабильности
        max_allowed = 0.99 / k
        x_scale = torch.where(x_norm_sq >= max_allowed, 
                            torch.sqrt(max_allowed / (x_norm_sq + 1e-8)),
                            torch.ones_like(x_norm_sq))
        y_scale = torch.where(y_norm_sq >= max_allowed,
                            torch.sqrt(max_allowed / (y_norm_sq + 1e-8)),
                            torch.ones_like(y_norm_sq))
        
        # Масштабируем векторы, чтобы они находились внутри диска
        x = x * x_scale.unsqueeze(-1)
        y = y * y_scale.unsqueeze(-1)
        
        # Пересчитываем квадраты норм
        x_norm_sq = torch.sum(x * x, dim=-1)
        y_norm_sq = torch.sum(y * y, dim=-1)
    
    # Вычисляем конформные факторы
    conformal_factor_x = (1.0 - k * x_norm_sq)
    conformal_factor_y = (1.0 - k * y_norm_sq)
    
    # Вычисляем евклидово скалярное произведение
    euclidean_inner = torch.sum(x * y, dim=-1)
    
    # Вычисляем гиперболическое скалярное произведение
    numerator = 4 * conformal_factor_x * conformal_factor_y * euclidean_inner
    denominator = (conformal_factor_x + conformal_factor_y) ** 2 + 1e-8
    
    return numerator / denominator

def extract_file_paths(diff_string: Optional[str]) -> Set[str]:
    """Извлекает уникальные пути файлов из строки diff."""
    if not isinstance(diff_string, str): return set()
    matches = re.findall(r'^ppp b /(.*?)<nl>', diff_string, re.MULTILINE)
    return {match.strip() for match in matches if match.strip() != '/dev/null'}

def find_lowest_common_ancestor(paths: Set[str]) -> str:
    """Находит LCA для набора путей."""
    if not paths: return ""
    split_paths = [[comp for comp in path.replace("\\", "/").split('/') if comp] for path in paths]
    if not split_paths or any(not p for p in split_paths): return ""
    min_len = min(len(p) for p in split_paths)
    lca_components = []
    for i in range(min_len):
        current_component = split_paths[0][i]
        if all(p[i] == current_component for p in split_paths): lca_components.append(current_component)
        else: break
    return "/".join(lca_components)

def get_heuristic_hyperbolic_embedding(file_path, alpha=0.45, beta=0.5, gamma=0.1, 
                                      manifold=poincare_ball, eps=1e-8,
                                      angle_hash_modulo=10000, embedding_dim=10, 
                                      dtype=DEFAULT_DTYPE):
    """
    Создает эвристический гиперболический вектор-эмбеддинг на основе глубины пути в файловой системе.
    Чем глубже файл в структуре директорий, тем ближе он к границе диска Пуанкаре.
    
    Args:
        file_path: Путь к файлу
        alpha: Коэффициент масштабирования (устаревший параметр)
        beta: Коэффициент экспоненциального масштабирования (новый параметр)
        gamma: Смещение глубины (новый параметр)
        manifold: Многообразие Пуанкаре
        eps: Малое значение для предотвращения числовой нестабильности
        angle_hash_modulo: Модуль для хэширования углов
        embedding_dim: Размерность эмбеддинга
        dtype: Тип данных
        
    Returns:
        Тензор PyTorch с гиперболическим эмбеддингом
    """
    # Удаляем любые ведущие и завершающие слэши
    normalized_path = file_path.strip('/')
    
    # Выполняем случайное размещение вектора в евклидовом пространстве (на единичной сфере)
    rng = np.random.RandomState(hash(normalized_path) % 2**32)
    
    # Создаем случайный вектор на единичной сфере
    # ИСПРАВЛЕНО: Используем embedding_dim вместо manifold.ndim
    v = rng.randn(embedding_dim)
    v = v / (np.linalg.norm(v) + eps)  # Нормализуем к единичной норме
    
    # Определяем глубину как количество компонентов в пути
    depth = len(normalized_path.split('/'))
    
        # Новая формула: radius = 1 - math.exp(-beta * (depth + gamma))
    radius = 1.0 - math.exp(-beta * (depth + gamma))
        
    # Масштабируем вектор
    v = radius * v
    
    # Проверяем, что вектор находится внутри диска Пуанкаре
    v_norm = np.linalg.norm(v)
    if v_norm >= 1.0:
        v = v * (1.0 - eps) / v_norm  # Масштабируем, чтобы была норма < 1
    
    # Преобразуем в тензор PyTorch
    v_tensor = torch.tensor(v, dtype=dtype)
    
    # Проверяем принадлежность к многообразию
    try:
        manifold.assert_check_point_on_manifold(v_tensor)
    except ValueError as e:
        logging.warning(f"Вектор не на многообразии: {e}")
        # Проекция на многообразие
        v_tensor = manifold.projx(v_tensor)
    
    return v_tensor

def calculate_hyperbolic_centroid(
    vectors,
    weights: Optional[torch.Tensor] = None,
    manifold: geoopt.PoincareBall = poincare_ball,
    initial_point: Optional[torch.Tensor] = None,
    max_iterations: int = 75,
    lr: float = 0.5,
    convergence_eps: float = 1e-4,
    clip_threshold: float = 0.999,
    use_information_weights: bool = True,
    embedding_dim: Optional[int] = None  # Добавляем параметр с None по умолчанию
) -> torch.Tensor:
    """
    Вычисляет взвешенный гиперболический центроид на диске Пуанкаре.
    
    Args:
        vectors: Тензор или список тензоров входных векторов
        weights: Веса для векторов формы [N], по умолчанию равные веса
        manifold: Многообразие Пуанкаре
        initial_point: Начальная точка для оптимизации
        max_iterations: Максимальное число итераций
        lr: Скорость обучения
        convergence_eps: Порог сходимости
        clip_threshold: Порог для клиппинга, чтобы избежать границы диска
        use_information_weights: Использовать ли информационно-взвешенное суммирование
        embedding_dim: Размерность эмбеддинга (если задана)
        
    Returns:
        torch.Tensor: Гиперболический центроид формы [D]
    """
    # Проверяем, является ли vectors списком тензоров и преобразуем его в тензор
    if isinstance(vectors, list):
        if not vectors:  # Пустой список
            if embedding_dim is None:
                # Если размерность не указана, возвращаем нулевой вектор размерности 1
                return torch.zeros(1, dtype=DEFAULT_DTYPE)
            else:
                # Если размерность указана, возвращаем нулевой вектор этой размерности
                return torch.zeros(embedding_dim, dtype=DEFAULT_DTYPE)
        try:
            vectors = torch.stack(vectors).to(DEFAULT_DTYPE)
        except:
            logging.error(f"Не удалось преобразовать список в тензор: {vectors}")
            # Возвращаем нулевой вектор нужной размерности
            if embedding_dim is not None:
                return torch.zeros(embedding_dim, dtype=DEFAULT_DTYPE)
            elif len(vectors) > 0 and hasattr(vectors[0], 'shape'):
                return torch.zeros(vectors[0].shape, dtype=DEFAULT_DTYPE)
            else:
                return torch.zeros(1, dtype=DEFAULT_DTYPE)
    
    # Если embedding_dim задан, используем его для валидации
    if embedding_dim is not None and vectors.shape[1] != embedding_dim:
        logging.warning(f"Размерность векторов ({vectors.shape[1]}) не соответствует заданной ({embedding_dim})")
    
    n_points, dim = vectors.shape
    
    # Проверка наличия параметров и установка значений по умолчанию
    if weights is None:
        weights = torch.ones(n_points, device=vectors.device, dtype=vectors.dtype) / n_points
    else:
        if weights.shape[0] != n_points:
            raise ValueError(f"Число весов ({weights.shape[0]}) не соответствует числу векторов ({n_points})")
        
        # Если информационные веса используются, применяем softmax для нормализации
        if use_information_weights:
            # Применяем softmax для получения распределения вероятностей
            # Это обеспечивает более чувствительное различение между весами
            weights = torch.softmax(weights, dim=0)
        else:
            # Стандартная нормализация
            weights = weights / weights.sum()
    
    # Если нет точек, возвращаем нулевой вектор (центр диска)
    if n_points == 0:
        return torch.zeros(dim, device=vectors.device, dtype=vectors.dtype)
    
    # Если только одна точка, возвращаем её
    if n_points == 1:
        return vectors[0].clone()
    
    # Инициализация начальной точки
    if initial_point is None:
        # Евклидово среднее взвешенное должно быть внутри диска
        euclidean_centroid = torch.sum(vectors * weights.unsqueeze(1), dim=0)
        norm = torch.norm(euclidean_centroid, p=2)
        
        # Проецируем на диск Пуанкаре, если необходимо
        if norm >= 1.0:
            euclidean_centroid = euclidean_centroid * (clip_threshold / norm)
            
        current_point = euclidean_centroid
    else:
        current_point = initial_point.clone()
    
    # Убеждаемся, что current_point требует градиента
    current_point = current_point.clone().detach().requires_grad_(True)
    
    # Оптимизация
    for i in range(max_iterations):
        # Вычисляем расстояния от текущей точки до всех векторов
        distances = manifold.dist(current_point, vectors)
        
        # Вычисляем взвешенную сумму квадратов расстояний
        loss = torch.sum(weights * distances.pow(2))
        
        # Выполняем шаг градиентного спуска
        # Используем retain_graph=True, чтобы предотвратить ошибку повторного повторного использования графа
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            if current_point.grad is None:
                break
                
            # Шаг в направлении отрицательного градиента
            current_point = current_point - lr * current_point.grad
            
            # Проецируем обратно на диск Пуанкаре и готовим к следующему шагу
            norm = torch.norm(current_point, p=2)
            if norm >= 1.0:
                current_point = current_point * (clip_threshold / norm)

            # «Отрываем» тензор от графа и снова включаем градиенты
            current_point = current_point.detach().requires_grad_(True)

            # Проверка сходимости
            if i > 0 and abs(loss.item() - prev_loss) < convergence_eps:
                break

            prev_loss = loss.item()
    
    # Очищаем вычислительный граф и возвращаем результат
    return current_point.detach()

# Добавить после функции calculate_hyperbolic_centroid
def batch_calculate_hyperbolic_centroids(file_paths_batch, changed_lines=None, alpha=0.45, beta=0.5, gamma=0.1, 
                                        manifold=poincare_ball, batch_size=32, use_weights=True, 
                                        max_samples=1000, aggregation="mean"):
    """
    Вычисляет гиперболические центроиды для пакета путей к файлам.
    
    Args:
        file_paths_batch: Пакет списков путей к файлам
        changed_lines: Списки измененных строк для файлов (для информационных весов)
        alpha: Коэффициент масштабирования (устаревший параметр)
        beta: Коэффициент экспоненциального масштабирования (новый параметр)
        gamma: Смещение глубины (новый параметр)
        manifold: Многообразие Пуанкаре
        batch_size: Размер пакета для обработки
        use_weights: Использовать ли веса при вычислении центроида
        max_samples: Максимальное количество примеров для обработки
        aggregation: Метод агрегации ("mean" или "weighted_mean")
        
    Returns:
        Список тензоров центроидов, по одному для каждого списка путей к файлам
    """
    centroids = []
    
    for i, file_paths in enumerate(file_paths_batch):
        if not file_paths:
            # Если нет путей, создаем нулевой вектор
            centroid = torch.zeros(manifold.dim, dtype=DEFAULT_DTYPE)
            # Проверяем принадлежность нулевого вектора многообразию (для надёжности)
            try:
                manifold.assert_check_point_on_manifold(centroid)
            except ValueError:
                # Если нулевой вектор вне многообразия, то проецируем его
                centroid = manifold.projx(centroid)
            centroids.append(centroid)
            continue
        
        # Ограничиваем количество путей для эффективности
        if max_samples > 0 and len(file_paths) > max_samples:
            indices = np.random.choice(len(file_paths), size=max_samples, replace=False)
            file_paths = [file_paths[idx] for idx in indices]
            if changed_lines is not None and i < len(changed_lines):
                changed_lines[i] = [changed_lines[i][idx] for idx in indices if idx < len(changed_lines[i])]
        
                weights = None
        if use_weights and changed_lines is not None and i < len(changed_lines) and changed_lines[i]:
            # Используем информацию о количестве измененных строк как веса
            weights = [max(1, lines_count) for lines_count in changed_lines[i]]
            
            # Нормализуем веса, чтобы их сумма была равна 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                
        # Получаем эмбеддинги для путей к файлам
        embeddings = []
                embedding_dim = manifold.dim
        dtype = DEFAULT_DTYPE
                for path in file_paths:
                        embedding = get_heuristic_hyperbolic_embedding(
                path, 
                alpha=alpha, 
                beta=beta, 
                gamma=gamma, 
                manifold=manifold,
                embedding_dim=embedding_dim,
                dtype=dtype
            )
                        embeddings.append(embedding)
        
        # Объединяем эмбеддинги в пакетный тензор
        embeddings_tensor = torch.stack(embeddings)
        
        # Вычисляем центроид
                centroid = calculate_hyperbolic_centroid(
            embeddings_tensor, 
            weights=weights, 
            manifold=manifold,
            embedding_dim=embedding_dim # <-- Добавлено
        )
                centroids.append(centroid)
    
    return centroids

# ==============================================================================
# 3. ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ
# ==============================================================================

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
            except (ValueError, TypeError): # Оставляем на случай других проблем
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
        diffs, messages, diff_ids, file_paths_list,
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
                train_val_diffs, train_val_msgs, train_val_ids, train_val_paths,
                train_size=relative_train_size, random_state=random_state)
                    else: 
             train_diffs, train_msgs, train_ids, train_paths = [], [], [], [] # Добавили train_paths
             val_diffs, val_msgs, val_ids, val_paths = train_val_diffs, train_val_msgs, train_val_ids, train_val_paths # Добавили val_paths
    else:
        train_diffs, val_diffs, train_msgs, val_msgs, train_ids, val_ids, train_paths, val_paths = [], [], [], [], [], [], [], [] # Добавили paths
        
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
    centroid_clip_threshold = config.get("centroid_clip_threshold", 0.999)
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
                embedding_dim=embedding_dim, # <-- Передаем размерность
                max_iterations=centroid_max_iter,
                lr=centroid_lr,
                convergence_eps=centroid_eps, 
                clip_threshold=centroid_clip_threshold
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
        torch.save(all_centroids, output_path)
        logging.info(f"Структурные эмбеддинги (центроиды) сохранены в {output_path}")
        return True
    except Exception as e:
        logging.error(f"Ошибка сохранения тензора структурных центроидов: {e}", exc_info=True)
        return False
    
def prepare_all_data(config: dict) -> bool:
    """Оркестрирует весь процесс подготовки данных, читая предобработанный JSONL."""
    logging.info("--- Шаг 0: Проверка и подготовка данных для обучения --- ") # Используем logging
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
    logging.info("\n--- Подготовка семантических эмбеддингов ---")
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
    
    logging.info("--- Шаг 0: Подготовка данных завершена успешно --- ")
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
        
                # Оставляем diffs под ключом question_diffs на случай, если где-то используются
        if 'diffs' in split_data:
            split_data['question_diffs'] = split_data['diffs'] 
        else:
            logging.warning(f"Ключ 'diffs' не найден в {file_path}")
            split_data['question_diffs'] = [""] * num_entries
            
        # Добавляем file_paths под ключом question_file_paths
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
    tokenized_dataset = None
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
            'question_file_paths': dataset_dict['question_file_paths']
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
            model_inputs["question_diffs"] = examples["question_diffs"]
            model_inputs["question_file_paths"] = examples["question_file_paths"]
                        return model_inputs
            
        tokenized_dataset = dataset.map(
            tokenize_fn, 
            batched=True, 
            remove_columns=['input_text', 'target_text'] 
            # Не удаляем question_diffs и question_file_paths, они останутся
        )
        # Указываем формат torch для основных колонок
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        # Колонки question_diffs и question_file_paths остаются в формате Python list
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Ошибка при токенизации датасета: {e}", exc_info=True) # Добавляем exc_info
        # import traceback; traceback.print_exc()
        return None
    
class CustomDataCollatorWithPaths(DataCollatorForSeq2Seq): # Переименовано
    """
    Кастомный Data Collator, который сохраняет колонку 'question_file_paths'.
    """
    def __call__(self, features, return_tensors=None):
                # Сохраняем diffs перед вызовом родительского collate (если нужны)
        question_diffs = [feature.pop("question_diffs", None) for feature in features]
        # Сохраняем file_paths 
        question_file_paths = [feature.pop("question_file_paths", []) for feature in features]
                
        # Вызываем стандартный collate для остальных полей (input_ids, attention_mask, labels)
        batch = super().__call__(features, return_tensors)
        
                # Добавляем diffs обратно в батч (если нужны)
        batch["question_diffs"] = question_diffs
        # Добавляем file_paths обратно в батч
        batch["question_file_paths"] = question_file_paths
                return batch

class RerankingSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Инициализируем хранилища для сбора метрик
        self.topk_metrics = {
            'baseline': {1: [], 3: [], 5: []},
            'rerank': {1: [], 3: [], 5: []}
        }
        self.mixing_scores_data = []
        
        # Проверяем наличие модуля визуализации
        self.viz_utils_available = _viz_utils_available
        
        # Создаем директории для логов в prepared_data_dir
        base_dir = CONFIG.get("prepared_data_dir", ".")
        self.logs_dir = os.path.join(base_dir, "logs")
        self.figures_dir = os.path.join(base_dir, "figures")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Расширенная evaluate: помимо стандартных метрик вычисляем Top‑k accuracy и сохраняем визуализации."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Расчёт Top-k accuracy перенесён в on_train_end для экономии времени.
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Переопределяем compute_loss для передачи question_file_paths в модель.
        """
                # question_diffs = inputs.pop("question_diffs", None) # Старое
        question_file_paths = inputs.pop("question_file_paths", None) 
                labels = inputs.get("labels")
                outputs = model(**inputs, question_file_paths=question_file_paths, return_dict=True)
                loss = outputs.loss 
        if loss is not None and loss.dim() > 0: loss = loss.mean()

                curvature_reg_weight = 0.1  # Вес для регуляризации кривизны (можно настроить)
        initial_curvature = 1.0  # Начальное значение кривизны (обычно 1.0)
        
        try:
            # Ищем параметр кривизны в poincare_ball, сохраненный в trainer
            if hasattr(self, 'poincare_ball'):
                curvature_param = None
                # Проверяем разные возможные имена параметра кривизны
                if hasattr(self.poincare_ball, 'c') and isinstance(self.poincare_ball.c, torch.nn.Parameter):
                    curvature_param = self.poincare_ball.c
                elif hasattr(self.poincare_ball, 'k') and isinstance(self.poincare_ball.k, torch.nn.Parameter):
                    # Используем -k для кривизны, если параметр называется k
                    curvature_param = -self.poincare_ball.k 
                
                if curvature_param is not None:
                    # L2 регуляризация: λ * (c - c₀)² или λ * (-k - c₀)²
                    # Приводим initial_curvature к типу и устройству curvature_param
                    initial_curvature_tensor = torch.tensor(initial_curvature, dtype=curvature_param.dtype, device=curvature_param.device)
                    curvature_loss = curvature_reg_weight * (curvature_param - initial_curvature_tensor).pow(2)
                    # Добавляем к основной потере
                    loss = loss + curvature_loss
                    logging.debug(f"Добавлена регуляризация кривизны: {curvature_loss.item():.6f}, текущее значение={curvature_param.item():.6f}")
            else:
                 logging.warning("Атрибут poincare_ball не найден в trainer для регуляризации кривизны.")
        except Exception as e:
            logging.warning(f"Ошибка при вычислении регуляризации кривизны: {e}")
                        curriculum_weight = CONFIG.get("curriculum_weight", 0.05)
        if curriculum_weight > 0:
            try:
                # Ищем rerank_weight в модели
                if hasattr(model, "rag") and hasattr(model.rag, "retriever") and hasattr(model.rag.retriever, "rerank_weight"):
                    rw = model.rag.retriever.rerank_weight
                    # Гарантируем, что это тензор
                    if not isinstance(rw, torch.Tensor):
                        rw = torch.tensor(float(rw), device=loss.device)
                    # Применяем регуляризацию (lambda - 0.5)²
                    target_val = 0.5
                    curriculum_loss = curriculum_weight * (rw - target_val) ** 2
                    loss = loss + curriculum_loss
                    logging.debug(f"Добавлена curriculum-регуляризация с весом {curriculum_weight}, текущий rerank_weight={rw.item():.4f}")
            except Exception as e:
                logging.warning(f"Ошибка при вычислении curriculum-регуляризации: {e}")
                        spearman_weight = CONFIG.get("spearman_weight", 0.0) # Используем 0.0 по умолчанию, если не задано
        if spearman_weight > 0 and question_file_paths is not None:
            try:
                # Получаем эмбеддинги вопроса (могут быть разного вида в зависимости от модели)
                question_embeddings = None
                # Пытаемся получить эмбеддинги из разных возможных мест outputs
                if hasattr(outputs, "question_hidden_states"): # RerankingRagRetriever может добавлять это
                     question_embeddings = outputs.question_hidden_states
                elif hasattr(outputs, "question_encoder_last_hidden_state"): # Стандартный RAGModel
                    # Проверяем размерность тензора и его тип
                    if outputs.question_encoder_last_hidden_state is None:
                        logging.warning("question_encoder_last_hidden_state is None, skipping Spearman regularization")
                    elif not isinstance(outputs.question_encoder_last_hidden_state, torch.Tensor):
                        logging.warning(f"question_encoder_last_hidden_state is not a tensor: {type(outputs.question_encoder_last_hidden_state)}")
                    elif outputs.question_encoder_last_hidden_state.dim() == 3:
                        # Тензор имеет 3 измерения [batch_size, sequence_length, hidden_size]
                        question_embeddings = outputs.question_encoder_last_hidden_state[:, 0, :]  # CLS токен
                    elif outputs.question_encoder_last_hidden_state.dim() == 2:
                        # Тензор имеет 2 измерения [batch_size, hidden_size] - уже извлечен CLS
                        question_embeddings = outputs.question_encoder_last_hidden_state
                    else:
                        logging.warning(f"Unexpected tensor shape for question_encoder_last_hidden_state: {outputs.question_encoder_last_hidden_state.shape}")
                elif hasattr(outputs, "encoder_last_hidden_state"):
                    if outputs.encoder_last_hidden_state is not None and outputs.encoder_last_hidden_state.dim() == 3:
                        question_embeddings = outputs.encoder_last_hidden_state[:, 0, :]
                elif hasattr(outputs, "encoder_outputs") and isinstance(outputs.encoder_outputs, ModelOutput) and hasattr(outputs.encoder_outputs, "last_hidden_state"):
                    # Если encoder_outputs это ModelOutput
                    if outputs.encoder_outputs.last_hidden_state is not None and outputs.encoder_outputs.last_hidden_state.dim() == 3:
                        question_embeddings = outputs.encoder_outputs.last_hidden_state[:, 0, :]
                elif isinstance(outputs.encoder_outputs, tuple) and len(outputs.encoder_outputs) > 0 and isinstance(outputs.encoder_outputs[0], torch.Tensor):
                     # Если encoder_outputs это кортеж
                     if outputs.encoder_outputs[0].dim() == 3:
                         question_embeddings = outputs.encoder_outputs[0][:, 0, :]

                if question_embeddings is not None:
                    question_norms = torch.norm(question_embeddings, p=2, dim=1)

                    batch_size = question_norms.shape[0]
                    depth_values = []
                    valid_depth_indices = [] # Индексы с ненулевыми путями
                    for i, paths in enumerate(question_file_paths):
                        if paths:
                            depths = [len(path.strip('/').split('/')) for path in paths]
                            avg_depth = sum(depths) / len(depths) if depths else 0.0
                            depth_values.append(avg_depth)
                            if avg_depth > 0: # Считаем только если глубина > 0
                                valid_depth_indices.append(i)
                        else:
                            depth_values.append(0.0) # Глубина 0 для пустых путей

                    if len(valid_depth_indices) > 1: # Нужно хотя бы 2 точки для корреляции
                        path_depths = torch.tensor(depth_values, device=question_norms.device, dtype=question_norms.dtype)

                        # Берем только валидные нормы и глубины
                        valid_norms = question_norms[valid_depth_indices].detach().cpu().numpy()
                        valid_depths = path_depths[valid_depth_indices].detach().cpu().numpy()

                        # Вычисляем корреляцию Спирмена
                        correlation, p_value = spearmanr(valid_norms, valid_depths)

                        # Проверяем на NaN (может возникнуть, если все глубины/нормы одинаковы)
                        if not np.isnan(correlation):
                            # Цель: положительная корреляция (глубже -> больше норма)
                            # Минимизируем -(correlation), т.е. максимизируем correlation
                            spearman_loss = -correlation
                            # Преобразуем обратно в тензор для добавления к loss
                            spearman_loss_tensor = torch.tensor(spearman_loss, device=loss.device, dtype=loss.dtype)
                            loss = loss + spearman_weight * spearman_loss_tensor
                            logging.debug(f"Добавлена Spearman-регуляризация: {-spearman_loss:.4f} (p={p_value:.3f}) с весом {spearman_weight}")
                        else:
                            logging.debug("Spearman correlation is NaN, skipping regularization.")
                    else:
                         logging.debug("Недостаточно данных (>1) для Spearman регуляризации в этом батче.")
                else:
                    logging.warning("Не удалось извлечь question_embeddings из outputs для Spearman-регуляризации.")

            except Exception as e:
                 logging.warning(f"Ошибка при вычислении Spearman-регуляризации: {e}", exc_info=True) # Добавлено exc_info=True
                return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Создаёт оптимизатор с разными learning rates для гиперболических и обычных параметров.
        """
        if self.optimizer is not None and self.lr_scheduler is not None:
            return  # Оптимизатор и планировщик уже созданы

        # Шаг 1. Сбор гиперболических параметров (манифольд + обучаемые скаляры)
        hyperbolic_params = []
        main_params = []
        
                # 1. Определяем, какой параметр кривизны используется в этой версии geoopt
        curv_param = None
        curv_param_name = None
        
        # Проверяем все возможные имена параметров кривизны в _parameters
        if 'isp_c' in poincare_ball._parameters:       # geoopt 0.5.0
            curv_param_name = 'isp_c'
            curv_param = poincare_ball._parameters['isp_c']
            logging.info(f"Обнаружен параметр кривизны 'isp_c' (geoopt 0.5.0+)")
        elif 'c' in poincare_ball._parameters:         # новая geoopt (≥ 0.5.0)
            curv_param_name = 'c'
            curv_param = poincare_ball._parameters['c']
            logging.info(f"Обнаружен параметр кривизны 'c' (новая версия geoopt)")
        elif 'k' in poincare_ball._parameters:         # старая geoopt (≤ 0.4.x)
            curv_param_name = 'k'
            curv_param = poincare_ball._parameters['k']
            logging.info(f"Обнаружен параметр кривизны 'k' (старая версия geoopt)")
        else:
            # Запасной вариант для других реализаций
            if hasattr(poincare_ball, 'c') and isinstance(poincare_ball.c, torch.nn.Parameter):
                curv_param = poincare_ball.c
                curv_param_name = 'c'
                logging.info(f"Параметр 'c' получен через атрибут (нестандартная реализация)")
            elif hasattr(poincare_ball, 'k') and isinstance(poincare_ball.k, torch.nn.Parameter):
                curv_param = poincare_ball.k
                curv_param_name = 'k'
                logging.info(f"Параметр 'k' получен через атрибут (нестандартная реализация)")
            else:
                logging.warning("Не найден параметр кривизны в PoincareBall. Гиперболические параметры не будут оптимизироваться.")
                # Сохраняем ссылку на объект многообразия для логирования в callbacks
                self.poincare_ball = poincare_ball
        
        # 2. Обеспечиваем, что параметр кривизны - leaf-тензор (если он найден)
        if curv_param is not None:
            if not curv_param.is_leaf:
                # Создаём leaf-копию параметра
                new_param = torch.nn.Parameter(
                    curv_param.detach().clone(), requires_grad=True
                )
                # Заменяем старый параметр новым в registry
                if curv_param_name:
                    poincare_ball._parameters[curv_param_name] = new_param
                curv_param = new_param
                logging.info(f"Создана leaf-копия параметра кривизны '{curv_param_name}'")
            
            # 3. Добавляем параметр кривизны в группу гиперболических параметров
            hyperbolic_params.append(curv_param)
            logging.info(f"Параметр кривизны '{curv_param_name}' добавлен в группу гиперболических: {curv_param}")
            
            # Сохраняем ссылки для доступа в callbacks
            self.curv_param_name = curv_param_name
            self.curv_param = curv_param
            self.poincare_ball = poincare_ball
                
        # 1.2 Находим и обрабатываем параметр веса переранжирования (rerank_weight)
        rerank_weight_params = []
        for module in self.model.modules():
            if hasattr(module, 'rerank_weight') and isinstance(module.rerank_weight, torch.nn.Parameter):
                if not module.rerank_weight.is_leaf:
                    # Создаем новый leaf-параметр
                    new_rerank = torch.nn.Parameter(
                        module.rerank_weight.detach().clone(),
                        requires_grad=True
                    )
                    # Заменяем старый параметр новым
                    module.rerank_weight = new_rerank
                    rerank_weight_params.append(new_rerank)
                    logging.info(f"Заменен non-leaf rerank_weight на leaf-параметр: {new_rerank}")
                else:
                    rerank_weight_params.append(module.rerank_weight)
                    logging.info(f"Найден leaf-параметр rerank_weight: {module.rerank_weight}")
        
        # Добавляем все найденные параметры веса в гиперболические
        hyperbolic_params.extend(rerank_weight_params)
        
                mixing_temp_params = []
        for module in self.model.modules():
            if hasattr(module, 'mixing_temperature') and isinstance(module.mixing_temperature, torch.nn.Parameter):
                if not module.mixing_temperature.is_leaf:
                    new_tau = torch.nn.Parameter(module.mixing_temperature.detach().clone(), requires_grad=True)
                    module.mixing_temperature = new_tau
                    mixing_temp_params.append(new_tau)
                    logging.info(f"Заменён non‑leaf mixing_temperature на leaf‑параметр: {new_tau}")
                else:
                    mixing_temp_params.append(module.mixing_temperature)
                    logging.info(f"Найден leaf‑параметр mixing_temperature: {module.mixing_temperature}")
        # Добавляем параметры температуры в гиперболическую группу ДО оптимизатора
        hyperbolic_params.extend(mixing_temp_params)
                
        # 1.3 Собираем основные параметры модели (исключая гиперболические)
        hyperbolic_param_ids = {id(p) for p in hyperbolic_params}
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in hyperbolic_param_ids:
                main_params.append(param)

        # Создаём группы параметров с разными learning rates
        hyperbolic_lr = 6e-5  # Повышенный lr для гиперболических параметров
        main_lr = self.args.learning_rate  # Стандартный lr для основных параметров
        
        optimizer_grouped_parameters = [
            {
                "params": hyperbolic_params,
                "lr": hyperbolic_lr,
                "weight_decay": 0.0,  # Обычно для гиперболических параметров не используется weight_decay
            },
            {
                "params": main_params,
                "lr": main_lr,
                "weight_decay": self.args.weight_decay,
            },
        ]
        
        logging.info(f"Создание оптимизатора с разными группами параметров:")
        logging.info(f" - Гиперболические параметры: {len(hyperbolic_params)} шт, lr={hyperbolic_lr}")
        logging.info(f" - Основные параметры: {len(main_params)} шт, lr={main_lr}")
        
        # Используем AdamW для обеих групп параметров
        from torch.optim import AdamW
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        # Создаём планировщик learning rate
        from transformers.optimization import get_scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            self.optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        
        # Сохраняем ссылку на poincare_ball в самом trainer для доступа из callback
        self.poincare_ball = poincare_ball
        logging.info(f"Сохранена ссылка на poincare_ball в trainer для логирования: {poincare_ball}")
        

        
        return self.optimizer, self.lr_scheduler

    # Примечание: Стандартный prediction_step в Seq2SeqTrainer уже вызывает model.generate,
    # если predict_with_generate=True. Нам нужно убедиться, что наш переопределенный
    # generate в RerankingRagSequenceForGeneration правильно принимает question_diffs через kwargs.
    # Возможно, потребуется переопределить и prediction_step, чтобы явно извлечь 
    # question_diffs из inputs и передать их в model.generate.

class RerankingRagRetriever:
    """
    Обертка над стандартным RagRetriever, добавляющая переранжирование
    на основе гиперболических структурных эмбеддингов.
    Имеет метод __call__ для совместимости с тем, как RAG вызывает ретривер.
    """
    def __init__(self,
                 base_rag_retriever: RagRetriever, # Стандартный ретривер HF
                 structural_embeddings: torch.Tensor, # Наш тензор
                 k_to_rerank: int = 20,
                 n_final: int = 5,
                 rerank_weight: float = 0.2,
                 alpha_depth_scaling: float = 0.1,
                 angle_hash_modulo: int = 10000,
                                  centroid_max_iterations: int = 75,
                 centroid_lr: float = 0.5,
                 centroid_convergence_eps: float = 1e-4,
                 centroid_clip_threshold: float = 0.999,
                 mixing_temperature: float = 1.0,  # <-- НОВЫЙ параметр для температурного смешивания
                 use_hyperbolic_inner_product: bool = True,  # <-- НОВЫЙ параметр для переключения на inner product
                                  hyperbolic_manifold: geoopt.PoincareBall = poincare_ball,
                 device: Optional[str] = None,
                 learnable_weight: bool = True):
        """
        Инициализация ретривера-обертки.
        """
        self.base_retriever = base_rag_retriever
        
                if not hasattr(self.base_retriever, 'config'):
            raise AttributeError("Переданный base_rag_retriever не имеет атрибута 'config'!")
                
        # Проверяем наличие остальных необходимых атрибутов
        if not hasattr(self.base_retriever, 'index') or \
           not hasattr(self.base_retriever.index, 'dataset'):
            raise ValueError("Переданный base_rag_retriever не имеет ожидаемых атрибутов index/dataset.")
            
        self.config = self.base_retriever.config # Теперь присвоение должно быть безопаснее
        if self.config.n_docs != n_final:
             print(f"Warning: base_retriever.config.n_docs ({self.config.n_docs}) != n_final ({n_final}). Устанавливаем config.n_docs={n_final}")
             self.config.n_docs = n_final
             
                resolved_device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(resolved_device)
                
        self.structural_embeddings = structural_embeddings.to(self.device)
                self.embedding_dim = self.structural_embeddings.shape[1] 
                self.k_to_rerank = k_to_rerank
        self.n_final = n_final
        # Заменяем статический вес на обучаемый параметр
        if learnable_weight:
            self.rerank_weight = torch.nn.Parameter(torch.tensor(rerank_weight, dtype=self.structural_embeddings.dtype))
        else:
            self.rerank_weight = rerank_weight

                self.mixing_temperature = torch.nn.Parameter(torch.tensor(mixing_temperature, dtype=self.structural_embeddings.dtype)) if learnable_weight else mixing_temperature
        self.use_hyperbolic_inner_product = use_hyperbolic_inner_product
        self.depth_beta = torch.nn.Parameter(torch.tensor(0.5, dtype=self.structural_embeddings.dtype)) if learnable_weight else 0.5
        self.depth_gamma = torch.nn.Parameter(torch.tensor(1.0, dtype=self.structural_embeddings.dtype)) if learnable_weight else 1.0
                    
        self.semantic_weight = 1.0 - self.rerank_weight  # Это надо будет изменить в forward
        self.manifold = hyperbolic_manifold
        self.dtype = self.structural_embeddings.dtype
        self.alpha_depth_scaling = alpha_depth_scaling
        self.angle_hash_modulo = angle_hash_modulo
                self.centroid_max_iterations = centroid_max_iterations
        self.centroid_lr = centroid_lr
        self.centroid_convergence_eps = centroid_convergence_eps
        self.centroid_clip_threshold = centroid_clip_threshold
                if not (0 <= self.rerank_weight <= 1): raise ValueError("rerank_weight...")
        
        # Размер базы знаний для проверки ID
        try:
            # Используем len(dataset) как наиболее надежный источник размера
            self.num_docs_in_kb = len(self.base_retriever.index.dataset)
        except Exception as e:
             print(f"Warning: Не удалось получить размер датасета из base_retriever.index.dataset ({e}). Используется размер structural_embeddings.")
             self.num_docs_in_kb = self.structural_embeddings.shape[0]
             \
        print(f"RerankingRagRetriever (обертка) инициализирован. Размер KB: {self.num_docs_in_kb}, Размерность эмб.: {self.embedding_dim}") # Добавлено info о размерности
        
        # Инициализация кэша центроидов
        self._init_centroid_cache()
    
    def _init_centroid_cache(self):
        """Инициализирует кэш для вычисленных центроидов."""
        
        @lru_cache(maxsize=5000)  # Кэшируем до 5000 уникальных наборов путей
        def _cached_centroid(paths_hash, embedding_dim, manifold_c):
            """
            Кэшированное вычисление центроида для заданного хэша путей.
            
            Args:
                paths_hash: Хэш кортежа путей файлов
                embedding_dim: Размерность эмбеддинга
                manifold_c: Значение кривизны многообразия (для различения разных многообразий)
                
            Returns:
                Центроид как тензор PyTorch
            """
            # Декодируем пути из хэша
            paths_tuple = self._path_hash_to_tuple[paths_hash]
            
            # Вычисляем индивидуальные эмбеддинги
            query_individual_embeddings = []
            for path_str in paths_tuple:
                h_point = get_heuristic_hyperbolic_embedding(
                    path_str, 
                    self.alpha_depth_scaling, 
                    self.angle_hash_modulo, 
                    embedding_dim=embedding_dim, 
                    dtype=self.dtype, 
                    manifold=self.manifold,
                    beta=self.depth_beta,  # <-- НОВОЕ: Передаем beta
                    gamma=self.depth_gamma  # <-- НОВОЕ: Передаем gamma
                )
                query_individual_embeddings.append(h_point)
            
            # Вычисляем центроид как обычно
            centroid = calculate_hyperbolic_centroid(
                query_individual_embeddings,
                embedding_dim=embedding_dim,
                max_iterations=self.centroid_max_iterations,
                lr=self.centroid_lr,
                eps=self.centroid_convergence_eps,
                clip_threshold=self.centroid_clip_threshold
            )
            
            return centroid
        
        self._cached_centroid = _cached_centroid
        self._path_hash_to_tuple = {}  # Словарь хэш -> кортеж путей
    
    def _get_paths_hash(self, paths):
        """
        Генерирует хэш для списка путей.
        
        Args:
            paths: Список путей файлов
            
        Returns:
            Хэш, который можно использовать как ключ для кэша
        """
        # Сортируем пути для обеспечения детерминизма
        sorted_paths = tuple(sorted(set(paths)))
        
        # Проверяем, есть ли этот кортеж в словаре
        paths_str = "|".join(sorted_paths)
        paths_hash = hashlib.md5(paths_str.encode('utf-8')).hexdigest()
        
        # Сохраняем соответствие хэш -> кортеж путей
        self._path_hash_to_tuple[paths_hash] = sorted_paths
        
        return paths_hash
    
    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix: Optional[str] = None,
        n_docs: Optional[int] = None, 
        return_tensors: Optional[str] = "pt",
        question_file_paths: Optional[List[List[str]]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Выполняет семантический поиск и переранжирование с использованием
        гиперболических структурных эмбеддингов на диске Пуанкаре.
        """
        effective_n_docs = n_docs if n_docs is not None else self.n_final
        question_hidden_states_torch = torch.from_numpy(question_hidden_states).to(self.device).to(self.dtype)
        batch_size = question_hidden_states_torch.shape[0]
        
        if question_file_paths is not None and len(question_file_paths) != batch_size:
            logging.warning("Размер question_file_paths не совпадает с batch_size. Игнорируем пути.")
            question_file_paths = None

        try:
            retrieved_doc_embeds_k_np, doc_ids_k_np, _ = self.base_retriever.retrieve(
                question_hidden_states, self.k_to_rerank 
            )
            retrieved_doc_ids_k = torch.from_numpy(doc_ids_k_np).to(self.device).long()
            retrieved_doc_embeds_k = torch.from_numpy(retrieved_doc_embeds_k_np).to(self.dtype).to(self.device)
            semantic_scores_k = torch.bmm(retrieved_doc_embeds_k, question_hidden_states_torch.unsqueeze(-1)).squeeze(-1)
        except Exception as e:
            print(f"Ошибка base_retriever.retrieve: {e}")
            return BatchEncoding({}, tensor_type=return_tensors)

                # ИЗМЕНЕНИЕ: Убираем batch_final_scores здесь, будем создавать напрямую из combined_scores
        batch_final_indices_in_k = torch.full((batch_size, effective_n_docs), -1, dtype=torch.long, device=self.device)
        
        # Подготовка путей для всего батча
        query_file_paths_batch = [question_file_paths[i] if question_file_paths and i < len(question_file_paths) else []
                                 for i in range(batch_size)]

        query_centroids = torch.zeros((batch_size, self.embedding_dim), dtype=self.dtype, device=self.device)
        
        start_time = time.time()
        
        # Используем кэш для вычисления центроидов
        for i in range(batch_size):
            current_paths = set(query_file_paths_batch[i]) if query_file_paths_batch[i] else set()
            if not current_paths:
                continue
                
            # Хэшируем пути для кэширования
            paths_hash = self._get_paths_hash(current_paths)
            manifold_c = float(self.manifold.c.item())
            
            # Получаем или вычисляем центроид с помощью lru_cache
            cached_centroid = self._cached_centroid(paths_hash, self.embedding_dim, manifold_c)
            query_centroids[i] = cached_centroid.to(self.device)
     
        centroid_compute_time = time.time() - start_time
        logging.debug(f"Время вычисления центроидов: {centroid_compute_time:.4f} сек.")
        
                start_time = time.time()
        
        # ИЗМЕНЕНИЕ: Создадим списки для хранения скоров и индексов
        all_final_scores = []
        # all_absolute_indices = [] # Этот список не нужен для final_doc_scores, но может понадобиться для final_doc_ids

        # Вычисляем структурные скоры в цикле по батчу, но с оптимизациями
        for i in range(batch_size):
            query_sem_scores = semantic_scores_k[i]
            query_cand_ids = retrieved_doc_ids_k[i]
            
            # Проверка валидности кандидатов
            if query_cand_ids.numel() == 0 or (query_cand_ids.numel()==1 and query_cand_ids[0] < 0):
                # ИЗМЕНЕНИЕ: Добавляем пустые значения, чтобы сохранить структуру батча
                all_final_scores.append(torch.zeros(effective_n_docs, dtype=self.dtype, device=self.device, requires_grad=False)) # Устанавливаем requires_grad=False для нулей
                batch_final_indices_in_k[i] = -1 # Заполняем индексы -1
                continue
                
            valid_cand_mask = query_cand_ids >= 0
            valid_cand_ids = query_cand_ids[valid_cand_mask]
            
            if valid_cand_ids.numel() > 0:
                valid_cand_ids = valid_cand_ids[(valid_cand_ids >= 0) & (valid_cand_ids < self.num_docs_in_kb)]

            if valid_cand_ids.numel() == 0:
                # ИЗМЕНЕНИЕ: Добавляем пустые значения
                all_final_scores.append(torch.zeros(effective_n_docs, dtype=self.dtype, device=self.device, requires_grad=False))
                batch_final_indices_in_k[i] = -1
                continue
                
            # Извлекаем предвычисленные центроиды кандидатов
            candidate_centroid_embs = self.structural_embeddings[valid_cand_ids]
            
            # Получаем центроид запроса
            query_centroid_emb = query_centroids[i].unsqueeze(0)
            
            # Векторизованное вычисление структурных скоров
            if self.use_hyperbolic_inner_product:
                # Используем гиперболическое скалярное произведение (больше = лучше)
                current_k = -self.manifold.c
                structural_scores = poincare_inner_product(query_centroid_emb, candidate_centroid_embs, k=current_k)
            else:
                # Используем отрицательное расстояние (ближе = лучше)
                structural_scores = -poincare_distance(query_centroid_emb, candidate_centroid_embs)
            
            # Получаем семантические скоры для валидных кандидатов
            query_sem_scores_valid = query_sem_scores[valid_cand_mask]
            
            # Нормализуем скоры
            def min_max_normalize(scores):
                if scores.numel() <= 1:
                    return torch.full_like(scores, 0.5)
                min_val, max_val = torch.min(scores), torch.max(scores)
                if max_val == min_val:
                    return torch.full_like(scores, 0.5)
                return (scores - min_val) / (max_val - min_val + 1e-8)
            
            sem_scores_norm = min_max_normalize(query_sem_scores_valid)
            struct_scores_norm = min_max_normalize(structural_scores)
            
            # Комбинируем семантические и структурные скоры с температурным смешиванием
            if isinstance(self.rerank_weight, torch.nn.Parameter):
                # Получаем температуру (гарантируем положительность)
                temperature = torch.abs(self.mixing_temperature) + 1e-6
                
                # Применяем температуру к нормализованным скорам
                sem_scores_temp = sem_scores_norm / temperature
                struct_scores_temp = struct_scores_norm / temperature
                
                # Применяем softmax для получения весов
                scores_stacked = torch.stack([sem_scores_temp, struct_scores_temp], dim=-1)
                score_weights = torch.softmax(scores_stacked, dim=-1)
                
                # Используем rerank_weight как параметр для балансировки между двумя компонентами
                semantic_weight = 1.0 - torch.clamp(self.rerank_weight, 0.0, 1.0)
                semantic_component = score_weights[:, 0] * sem_scores_norm
                structural_component = score_weights[:, 1] * struct_scores_norm
                
                # Итоговое комбинирование
                combined_scores = semantic_weight * semantic_component + self.rerank_weight * structural_component
            else:
                # Статический вариант (для совместимости)
                semantic_weight = 1.0 - self.rerank_weight
                combined_scores = (semantic_weight * sem_scores_norm + self.rerank_weight * struct_scores_norm)
            
            # Выбираем топ-n
            actual_n = min(effective_n_docs, combined_scores.shape[0])
            top_n_scores, top_n_relative_indices = torch.topk(combined_scores, k=actual_n, largest=True)
            
            # Конвертируем относительные индексы в абсолютные
            absolute_indices_in_original_k = torch.where(valid_cand_mask)[0][top_n_relative_indices]
            
            # ИЗМЕНЕНИЕ: Сохраняем скоры и индексы в списки вместо заполнения тензора
            # Расширяем до effective_n_docs, заполняя неиспользуемые места нулями/минус-единицами
            if actual_n < effective_n_docs:
                # Создаем паддинг без градиента
                padding_scores = torch.zeros(effective_n_docs - actual_n, dtype=self.dtype, device=self.device, requires_grad=False)
                padding_indices = torch.full((effective_n_docs - actual_n,), -1, dtype=torch.long, device=self.device)
                
                final_scores_for_batch = torch.cat([top_n_scores, padding_scores])
                final_indices_for_batch = torch.cat([absolute_indices_in_original_k, padding_indices])
            else:
                final_scores_for_batch = top_n_scores
                final_indices_for_batch = absolute_indices_in_original_k
                
            all_final_scores.append(final_scores_for_batch)
            # all_absolute_indices.append(final_indices_for_batch) # Не используется напрямую для скоров
            
            # Сохраняем индексы для сбора ID и эмбеддингов
            batch_final_indices_in_k[i] = final_indices_for_batch
        
        ranking_compute_time = time.time() - start_time
        logging.debug(f"Время переранжирования: {ranking_compute_time:.4f} сек.")
        
        # ИЗМЕНЕНИЕ: Формируем тензоры из списков, сохраняя град-связи
        if all_final_scores:
            final_doc_scores = torch.stack(all_final_scores)
        else:
            # Запасной вариант, если нет результатов
            final_doc_scores = torch.zeros((batch_size, effective_n_docs), dtype=self.dtype, device=self.device, requires_grad=False)
        
        # Шаг 3: Подготовка финального вывода BatchEncoding
        # 3.1 Собираем финальные ID и Эмбеддинги (топ-n)
        final_doc_ids = retrieved_doc_ids_k.gather(dim=1, index=batch_final_indices_in_k)
        final_doc_ids[batch_final_indices_in_k == -1] = -100 # Используем -100 как паддинг 
        
        expanded_indices = batch_final_indices_in_k.unsqueeze(-1).expand(-1, -1, retrieved_doc_embeds_k.shape[-1])
        final_doc_embeds = retrieved_doc_embeds_k.gather(dim=1, index=expanded_indices)
        final_doc_embeds[batch_final_indices_in_k == -1] = 0 
        
        # 3.2 Извлекаем тексты для финальных ID
        final_doc_ids_list = final_doc_ids.cpu().tolist()
        docs_list = []
        for ids_for_sample in final_doc_ids_list:
             valid_ids = [id_ for id_ in ids_for_sample if id_ >= 0]
             if valid_ids:
                 try: docs_list.append(self.base_retriever.index.dataset[valid_ids])
                 except IndexError: print(f"Ошибка индекса при извлечении документов: {valid_ids}"); docs_list.append({"title": [], "text": []})
             else: docs_list.append({"title": [], "text": []})
        docs = []
        for i in range(batch_size):
            sample_docs = docs_list[i]
            num_retrieved = len(sample_docs["title"])            
            docs.append({
                 "title": sample_docs["title"] + ["PAD"] * (effective_n_docs - num_retrieved),
                 "text": sample_docs["text"] + ["PAD"] * (effective_n_docs - num_retrieved),
            })

        # 3.3 Декодируем запрос (используем ID, переданные от RagModel)
        input_strings = self.base_retriever.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        
        # 3.4 Вызываем postprocess_docs для форматирования и токенизации
        generator_prefix = prefix if prefix is not None else self.config.generator.prefix
        context_input_ids, context_attention_mask = self.base_retriever.postprocess_docs(
             docs, input_strings, generator_prefix, 
             effective_n_docs, return_tensors=return_tensors
        )
        
        # 3.5 Собираем BatchEncoding
        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": final_doc_embeds, 
                "doc_ids": final_doc_ids, 
                "doc_scores": final_doc_scores, 
            },
            tensor_type=return_tensors,
        )

class RerankingRagSequenceForGeneration(RagSequenceForGeneration):
    """Подкласс RagSequenceForGeneration для передачи question_diffs ретриверу.
       Использует стандартную логику RagModel.forward после получения 
       результатов от кастомного ретривера."""
    
    # Используем стандартный __init__ от RagSequenceForGeneration
    # Ретривер будет заменен после инициализации

    # Переопределяем forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[ModelOutput] = None, 
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        context_input_ids: Optional[torch.LongTensor] = None, # Эти теперь будут из ретривера
        context_attention_mask: Optional[torch.LongTensor] = None,
        doc_scores: Optional[torch.FloatTensor] = None, # Этот тоже
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        exclude_bos_score: Optional[bool] = None,
        reduce_loss: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        n_docs: Optional[int] = None,
        question_file_paths: Optional[List[List[str]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], RetrievAugLMMarginOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        n_docs = n_docs if n_docs is not None else self.config.n_docs

                if encoder_outputs is None:
            if input_ids is None: raise ValueError("input_ids or encoder_outputs must be given")
            question_encoder_outputs = self.rag.question_encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states
            )
        else: question_encoder_outputs = encoder_outputs
        question_hidden_states = getattr(question_encoder_outputs, "pooler_output", question_encoder_outputs.last_hidden_state[:, 0, :])
        
                if not isinstance(self.rag.retriever, RerankingRagRetriever):
            raise TypeError(f"Expected RerankingRagRetriever, got {type(self.rag.retriever)}")
        
                retriever_outputs: BatchEncoding = self.rag.retriever(
            question_input_ids=input_ids.tolist(),
            question_hidden_states=question_hidden_states.detach().cpu().numpy(),
            n_docs=n_docs, 
            prefix=self.config.generator.prefix, 
            return_tensors="pt",
            question_file_paths=question_file_paths
        )
                
                        # model_kwargs = kwargs.copy()
        # model_kwargs.pop("question_diffs", None) 
                
        # Вызываем self.rag.forward (RagModel)
        rag_model_output = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            # Передаем данные из ретривера
            context_input_ids=retriever_outputs.get("context_input_ids"),
            context_attention_mask=retriever_outputs.get("context_attention_mask"),
            doc_scores=retriever_outputs.get("doc_scores"), 
            # Передаем остальные флаги и аргументы
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            exclude_bos_score=exclude_bos_score,
            reduce_loss=reduce_loss,
            labels=labels,
            n_docs=n_docs,
            return_dict=return_dict,
                        **kwargs, # `question_file_paths` не входит в kwargs, так как был явным аргументом
                    )
        
                return rag_model_output

        @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
                question_file_paths: Optional[List[List[str]]] = None, 
                context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        **kwargs, # Принимаем остальные стандартные аргументы generate
    ) -> Union[GenerateOutput, torch.LongTensor]:

                if question_file_paths is None and "question_file_paths" in kwargs:
            question_file_paths = kwargs.pop("question_file_paths")
        else: 
            kwargs.pop("question_file_paths", None)
        
        # 1. Получаем параметры генерации и извлекаем флаги
        generation_config = generation_config if generation_config is not None else self.generation_config
        kwargs["generation_config"] = generation_config 
        
                output_attentions = kwargs.get("output_attentions", generation_config.output_attentions)
        output_hidden_states = kwargs.get("output_hidden_states", generation_config.output_hidden_states)
        use_cache = kwargs.get("use_cache", generation_config.use_cache)
                
        # 2. Подготовка model_kwargs и обработка question_diffs
        # ... (код извлечения/удаления/добавления question_diffs в model_kwargs) ...

        # 3. Получаем n_docs
        # ... 
        n_docs = generation_config.n_docs if hasattr(generation_config, "n_docs") else self.config.n_docs

        # 4. Получаем question_hidden_states
        # ... 
                if input_ids is None: raise ValueError("'input_ids' must be provided for generation.")
        question_encoder_outputs = self.rag.question_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=True,
            # Флаги output_attentions/hidden_states не нужны для ретривера
        )
        # Используем pooler_output если есть, иначе CLS
        question_hidden_states = getattr(question_encoder_outputs, "pooler_output", question_encoder_outputs.last_hidden_state[:, 0, :])
        
        # 5. Вызов НАШЕГО ретривера
        # ... 
                retriever_outputs: BatchEncoding = self.rag.retriever(
            question_input_ids=input_ids.tolist(),
            question_hidden_states=question_hidden_states.detach().cpu().numpy(),
            n_docs=n_docs, 
            prefix=self.config.generator.prefix, 
            return_tensors="pt",
            question_file_paths=question_file_paths # Передаем diffs
        )
        
        # 6. Подготовка входов для генератора BART
        # ... 
                encoder_input_ids = retriever_outputs.get("context_input_ids")
        encoder_attention_mask = retriever_outputs.get("context_attention_mask")
        if encoder_input_ids is None or encoder_attention_mask is None:
            raise ValueError("Retriever did not return 'context_input_ids' or 'context_attention_mask'.")
        # Перемещаем на нужное устройство, если нужно (хотя ретривер должен возвращать на нем)
        encoder_input_ids = encoder_input_ids.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        
        # 7. Подготовка kwargs для генератора BART
        generator_kwargs = {
            "encoder_outputs": ModelOutput(last_hidden_state=self.rag.generator.model.encoder(encoder_input_ids, attention_mask=encoder_attention_mask)[0]),
            "attention_mask": encoder_attention_mask,
            # Передаем извлеченные флаги
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
        }
        # ... (код копирования bart_kwargs и bart_kwargs.update(generator_kwargs))
                # Копируем релевантные kwargs для BART генератора
        # (Можно сделать более полным, если нужно больше параметров)
        bart_kwargs = {
             k: kwargs[k] 
             for k in ["max_length", "min_length", "num_beams", "do_sample", 
                       "temperature", "top_k", "top_p", "repetition_penalty", 
                       "length_penalty", "early_stopping", "num_return_sequences"]
             if k in kwargs
        }
        # Добавляем generation_config, если его нет в kwargs, но он нужен BART
        if "generation_config" not in bart_kwargs:
            bart_kwargs["generation_config"] = generation_config
        # Добавляем специфичные для RAG encoder_outputs и attention_mask
        bart_kwargs.update(generator_kwargs) 
                
        # 8. Генерация последовательностей генератором BART
        output_sequences = self.rag.generator.generate(
            input_ids=None, # Т.к. передаем encoder_outputs
            **bart_kwargs
        )

        # 9. Переранжирование лучей / выбор лучшей последовательности
                batch_size = input_ids.shape[0]
        n_docs = self.config.n_docs
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        num_beams = kwargs.get("num_beams", 1) # Получаем num_beams
        effective_batch_size = batch_size * n_docs
        num_generated_sequences = output_sequences.shape[0]
        # Учитываем num_beams при расчете кандидатов
        num_candidates = num_generated_sequences // (batch_size * n_docs) 
        
        # Повторяем input_ids и attention_mask
        repeated_input_ids = input_ids.repeat_interleave(n_docs * num_candidates, dim=0)
        repeated_attention_mask = attention_mask.repeat_interleave(n_docs * num_candidates, dim=0)
        
        # Повторяем question_diffs
        repeated_question_diffs = None
        if question_file_paths:
            # Повторяем каждый diff n_docs * num_candidates раз
            repeated_question_diffs = [q for q in question_file_paths for _ in range(n_docs * num_candidates)]
                
                # repeated_question_file_paths = None # Старое
        # if question_diffs: # Старое
        #     repeated_question_file_paths = [q for q in question_diffs for _ in range(n_docs * num_candidates)] # Старое
        repeated_question_file_paths = None
        if question_file_paths:
            # Повторяем каждый список путей n_docs * num_candidates раз
            repeated_question_file_paths = [p for p in question_file_paths for _ in range(n_docs * num_candidates)]
                
        # Вызываем НАШ forward для скоринга
                forward_kwargs = {
             "input_ids": repeated_input_ids,
             "attention_mask": repeated_attention_mask,
             "labels": output_sequences, 
             "exclude_bos_score": True,
             "reduce_loss": False, 
             "output_attentions": output_attentions,
             "output_hidden_states": output_hidden_states,
             "use_cache": False, 
             "question_file_paths": repeated_question_file_paths,
             "return_dict": True,
             # Передаем данные из ретривера для forward
             "context_input_ids": retriever_outputs.get("context_input_ids").repeat_interleave(num_candidates, dim=0),
             "context_attention_mask": retriever_outputs.get("context_attention_mask").repeat_interleave(num_candidates, dim=0),
             "doc_scores": retriever_outputs.get("doc_scores").repeat_interleave(num_candidates, dim=0) 
        }
        outputs = self.forward(**forward_kwargs)
        
        # ... (Заглушка выбора лучших)
        # return output_sequences[:batch_size * kwargs.get("num_return_sequences", 1)]
                # Получаем лосс для каждой сгенерированной последовательности (NLL Loss)
        # outputs.loss должен иметь размер (batch_size * n_docs * num_candidates)
        # Где num_candidates = num_beams * num_return_sequences (если они есть)
        if outputs.loss is None:
            print("Warning: self.forward did not return loss, cannot perform beam selection. Returning all sequences.")
            return output_sequences # Возвращаем все как есть
            
        # Используем отрицательный лосс как скор (чем меньше лосс, тем выше скор)
        sequence_scores = -outputs.loss 
        
        # Меняем форму скоров: (batch_size, n_docs * num_candidates)
        sequence_scores = sequence_scores.view(batch_size, -1)
        
        # Определяем, сколько последовательностей вернуть для каждого примера в батче
        num_return_sequences_final = kwargs.get("num_return_sequences", 1)
        
        # Находим топ-N скоров и их индексы в пределах группы (n_docs * num_candidates)
        top_scores, top_indices = sequence_scores.topk(num_return_sequences_final, dim=1)
        
        # Меняем форму сгенерированных последовательностей: (batch_size, n_docs * num_candidates, seq_len)
        output_sequences_reshaped = output_sequences.view(batch_size, -1, output_sequences.shape[-1])
        
        # Собираем лучшие последовательности с помощью gather
        # top_indices нужно расширить для gather по последнему измерению (seq_len)
        final_sequences = output_sequences_reshaped.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, output_sequences.shape[-1])
        )
        
        # Возвращаем финальные последовательности в виде (batch_size * num_return_sequences, seq_len)
        return final_sequences.view(-1, output_sequences.shape[-1])
        
    
def initialize_reranking_rag_model(config: dict):
    # ... (код до инициализации финальной модели)
    
        rag_config, tokenizer, question_encoder, generator = None, None, None, None
    base_retriever, structural_embeddings_kb = None, None
    reranking_retriever_wrapper = None
    model = None
        
    # ... (проверки файлов)

    print(f"Иниц. RAG ({config['rag_model_name']}) с переранж. ...")
    # ... (вывод параметров)

    # 1. Загрузка стандартных компонентов
    try:
        print("Загрузка Config, Tokenizer, Encoder, Generator...")
        rag_config = RagConfig.from_pretrained(config['rag_model_name'], index_name="custom", n_docs=config['n_final'])
        tokenizer = RagTokenizer.from_pretrained(config['rag_model_name'])
        question_encoder = DPRQuestionEncoder.from_pretrained(config['rag_model_name'])
        generator = BartForConditionalGeneration.from_pretrained(config['rag_model_name'])
        print("Стандартные компоненты загружены.")
    except Exception as e: 
        print(f"Ошибка загрузки компонентов RAG: {e}"); return None, None
        
    # Проверка после try-блока
    if not all([rag_config, tokenizer, question_encoder, generator]):
        print("Критическая ошибка: Не удалось загрузить все базовые компоненты.")
        return None, None

    # 2. Инициализация СТАНДАРТНОГО RagRetriever
    print("Инициализация базового RagRetriever...")
    try:
        base_retriever = RagRetriever.from_pretrained(
            config['rag_model_name'], index_name="custom",
            passages_path=config['knowledge_base_dataset_path'], index_path=config['faiss_index_path']
        )
        # ... (проверка индекса)
        print(f"Базовый RagRetriever инициализирован (Размер KB: {len(base_retriever.index.dataset)}).")
    except Exception as e:
        print(f"Ошибка инициализации базового RagRetriever: {e}"); import traceback; traceback.print_exc(); return None, None
    
    if base_retriever is None: 
        print("Критическая ошибка: Базовый ретривер не был инициализирован.")
        return None, None

    # 3. Загрузка Структурных Эмбеддингов
    try:
                logging.info(f"Загрузка структурных эмбеддингов из {config['structural_embeddings_output_path']}...")
        structural_embeddings_kb = torch.load(config['structural_embeddings_output_path'], map_location='cpu')
        logging.info(f"Структурные эмбеддинги загружены, размер: {structural_embeddings_kb.shape}")
        
        # Проверка соответствия размера структурных эмб. размеру KB
        kb_size_retriever = len(base_retriever.index.dataset) # Размер КБ из ретривера
        if structural_embeddings_kb.shape[0] != kb_size_retriever:
             raise ValueError(f"Размер структурных эмбеддингов ({structural_embeddings_kb.shape[0]}) "
                              f"не совпадает с размером KB Dataset ({kb_size_retriever})! "
                              f"Убедитесь, что prepare_dataset.py отработал корректно и использовался тот же JSONL.")
        logging.info("Размер структурных эмбеддингов соответствует размеру KB.")
            except FileNotFoundError:
         logging.error(f"Файл структурных эмбеддингов не найден: {config['structural_embeddings_output_path']}. Запустите prepare_dataset.py.")
         return None, None
    except Exception as e: 
        logging.error(f"Ошибка загрузки/проверки стр. эмбеддингов: {e}"); 
        return None, None

    # 4. Инициализация НАШЕЙ ОБЕРТКИ RerankingRagRetriever
    print("Инициализация RerankingRagRetriever (обертки)...")
    try:
                reranking_retriever_wrapper = RerankingRagRetriever(
            base_rag_retriever=base_retriever,
            structural_embeddings=structural_embeddings_kb,
            k_to_rerank=config['k_to_rerank'], 
            n_final=config['n_final'], 
            rerank_weight=config['rerank_weight'],
            alpha_depth_scaling=config['alpha_depth_scaling'], 
            angle_hash_modulo=config['angle_hash_modulo'], 
            # Передаем новые параметры
            centroid_max_iterations=config.get('centroid_max_iterations', 50),
            centroid_lr=config.get('centroid_lr', 0.5),
            centroid_convergence_eps=config.get('centroid_convergence_eps', 1e-4),
            centroid_clip_threshold=config.get('centroid_clip_threshold', 0.999),
            device=config['device'],
            learnable_weight=True
        )
                print("RerankingRagRetriever (обертка) создан.")
    except Exception as e: 
        # ... (обработка ошибок) ...
        pass # Или return None, None

    if reranking_retriever_wrapper is None:
        print("Критическая ошибка: Ретривер-обертка не был инициализирован.")
        return None, None

    # 5. Инициализация Финальной Модели RAG с ЗАМЕНОЙ ретривера
    try:
        print("Сборка финальной модели RerankingRagSequenceForGeneration...")
                # Сначала создаем стандартную модель RagSequenceForGeneration
        model = RagSequenceForGeneration.from_pretrained(
             config['rag_model_name'],
             config=rag_config, # Передаем конфиг
             question_encoder=question_encoder, # Передаем компоненты
             generator=generator,
             retriever=base_retriever # Передаем СТАНДАРТНЫЙ ретривер для инициализации
        )
        # Теперь ЗАМЕНЯЕМ ретривер на нашу обертку
        model.rag.retriever = reranking_retriever_wrapper
        
                # Это решает проблему, когда оригинальный RAG все равно пытается вызвать
        # generator.generate с question_diffs, который пришел из Trainer
        original_generator_generate = model.rag.generator.generate
        
        def patched_generator_generate(*args, **kwargs):
                        if "question_file_paths" in kwargs:
                kwargs.pop("question_file_paths", None)
            if "question_diffs" in kwargs: # Добавляем удаление старого ключа
                kwargs.pop("question_diffs", None)
                        return original_generator_generate(*args, **kwargs)
        
        # Заменяем метод generate у внутреннего генератора
        print("Патчим метод generate у внутреннего генератора BART...")
        model.rag.generator.generate = patched_generator_generate
                
                
                # Это гарантирует, что Trainer найдет его и не будет ошибки NoneType
        try:
            # Пытаемся создать из конфигурации модели (должна быть RagConfig)
            if hasattr(model, 'config') and model.config is not None:
                 model.generation_config = GenerationConfig.from_model_config(model.config)
                 print(f"GenerationConfig явно установлен из model.config.")
            else:
                 raise ValueError("model.config is missing or None")
        except Exception as e:
            print(f"Не удалось установить GenerationConfig из model.config ({e}). Пытаемся загрузить из {config['rag_model_name']}...")
            try:
                # Запасной вариант: загружаем из имени базовой модели
                model.generation_config = GenerationConfig.from_pretrained(config['rag_model_name'])
                print(f"GenerationConfig явно установлен из {config['rag_model_name']}.")
            except Exception as e2:
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось установить GenerationConfig: {e2}")
                # Можно вернуть None или поднять исключение, если это критично
                # return None, None 
                        
        print("Кастомный ретривер успешно установлен.")
        
                # Это нужно для функции _pad_tensors_to_max_len в Seq2SeqTrainer
        if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
            pad_token_id_source = None
            if hasattr(generator, 'config') and generator.config.pad_token_id is not None:
                model.config.pad_token_id = generator.config.pad_token_id
                pad_token_id_source = "generator.config"
            elif hasattr(tokenizer, 'generator') and tokenizer.generator.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.generator.pad_token_id
                pad_token_id_source = "tokenizer.generator"
            elif hasattr(generator, 'config') and generator.config.eos_token_id is not None:
                 # Запасной вариант: использовать EOS как PAD
                 model.config.pad_token_id = generator.config.eos_token_id
                 pad_token_id_source = "generator.config.eos_token_id (fallback)"
                 print(f"Warning: pad_token_id не найден. Установлен model.config.pad_token_id = eos_token_id ({model.config.pad_token_id})")
            
            if pad_token_id_source:
                 print(f"Установлен model.config.pad_token_id = {model.config.pad_token_id} из {pad_token_id_source}")
            else:
                 print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось определить pad_token_id для model.config. Паддинг в Trainer может не работать.")
                 # return None, None # Можно раскомментировать, если это фатально
                             
        model.to(config['device'])
        print("Модель RerankingRagSequenceForGeneration успешно инициализирована.")
        return tokenizer, model
    except Exception as e:
        print(f"Ошибка инициализации/замены ретривера в RagSequenceForGeneration: {e}")
        import traceback; traceback.print_exc(); return None, None
        

def postprocess_text(preds, labels):
    """Очистка текста и добавление переносов для ROUGE/METEOR."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds, tokenizer=None):
    """
    Вычисляет метрики для оценки качества генерации.
    
    Args:
        eval_preds: Кортеж (предсказания, метки)
        tokenizer: Токенизатор для декодирования последовательностей (опционально)
    
    Returns:
        Словарь с метриками
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
        # Получаем ID паддинга из токенизатора
    pad_token_id = None
    
    if tokenizer is not None:
        # Если токенизатор передан как аргумент (предпочтительно)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                logging.warning("Не удалось определить pad_token_id или eos_token_id из переданного tokenizer.")
    else:
        # Fallback: пытаемся получить глобальный токенизатор из области видимости
        try:
            # Глобальный токенизатор должен быть RagTokenizer с полем generator
            gen_tokenizer = tokenizer.generator
            pad_token_id = gen_tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = gen_tokenizer.eos_token_id
        except (NameError, AttributeError) as e:
            logging.warning(f"Не удалось получить токенизатор: {e}")
    
    # Если не удалось получить pad_token_id, используем 0 или 1 как резервные значения
    if pad_token_id is None:
        pad_token_id = 0
        logging.warning("Используем pad_token_id=0 по умолчанию для декодирования.")
    
    # Приводим тензоры к NumPy на CPU (если они еще не там)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Заменяем -100 на pad_token_id в предсказаниях и метках
    preds = np.where(preds == -100, pad_token_id, preds)
    labels = np.where(labels == -100, pad_token_id, labels)
    
    # Декодируем последовательности
    decoded_preds = []
    decoded_labels = []
    
    if tokenizer is not None:
        # Используем переданный токенизатор
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Ошибка декодирования с переданным токенизатором: {e}")
            return {}
    else:
        # Fallback с глобальным токенизатором
        try:
            decoded_preds = gen_tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = gen_tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Ошибка декодирования с глобальным токенизатором: {e}")
            return {}
    
    # Постобработка для метрик
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # Инициализируем словарь метрик
    metric_results = {}
    
    # Вычисляем метрики
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    
    try: 
        bleu = evaluate.load("bleu")
        result_bleu = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        metric_results["bleu"] = result_bleu["bleu"]
    except Exception as e: 
        logging.error(f"Ошибка вычисления BLEU: {e}")
    
    try:
        rouge = evaluate.load('rouge')
        result_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Добавляем ROUGE-L и усредненные значения
        metric_results.update({k: v * 100 for k, v in result_rouge.items()}) # Используем ROUGE score * 100
    except Exception as e: 
        logging.error(f"Ошибка вычисления ROUGE: {e}")
        
    try:
        meteor = evaluate.load('meteor')
        result_meteor = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        metric_results["meteor"] = result_meteor["meteor"]
    except Exception as e: 
        logging.error(f"Ошибка вычисления METEOR: {e}")

    # Округление метрик
    metric_results = {k: round(v, 4) for k, v in metric_results.items()}
    
    # Явное логирование вычисленных метрик
    logging.info(f"Calculated metrics: {metric_results}")
    
    # Добавляем значение кривизны в метрики
    try:
        # Пытаемся получить значение кривизны из poincare_ball
        if hasattr(poincare_ball, 'k'):
            k_value = poincare_ball.k.item() if isinstance(poincare_ball.k, torch.Tensor) else poincare_ball.k
            metric_results["curvature_k"] = round(k_value, 6)
        if hasattr(poincare_ball, 'c'):
            c_value = poincare_ball.c.item() if isinstance(poincare_ball.c, torch.Tensor) else poincare_ball.c
            metric_results["curvature_c"] = round(c_value, 6)
    except Exception as e:
        logging.warning(f"Не удалось добавить значение кривизны в метрики: {e}")
    
    return metric_results

class LoggingCallback(EarlyStoppingCallback): # Наследуем от EarlyStopping, чтобы не потерять его
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                base_dir = CONFIG.get("prepared_data_dir", ".")
        figures_base_dir = CONFIG.get("figures_dir", os.path.join(base_dir, "figures")) # Новая настройка
        logs_base_dir = CONFIG.get("logs_dir", os.path.join(base_dir, "logs")) # Новая настройка

        self.logs_dir = logs_base_dir
        self.figures_dir = figures_base_dir
                os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.viz_utils_available = _viz_utils_available
        self.current_epoch = 0
        self.log_centroids_every = CONFIG.get("log_centroids_every", 1) # Параметр из CONFIG

    def set_trainer(self, trainer):
        # Сохраняем ссылку на Trainer для использования в методах обратного вызова
        self.trainer = trainer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        if metrics is not None and state.is_world_process_zero:
             metrics_rounded = {k: round(v, 4) if isinstance(v, float) else v for k,v in metrics.items()}
             logging.info(f"Evaluation metrics: {metrics_rounded}")
                          try:
                 model = kwargs.get("model", None)
                 # Используем state.epoch, так как он обновляется до on_evaluate
                 current_eval_epoch = int(state.epoch) if state.epoch is not None else self.current_epoch
                 if model is not None and self.viz_utils_available:
                     viz_utils.save_model_stats(
                         model=model,
                         epoch=current_eval_epoch,
                         output_dir=os.path.join(self.logs_dir, "model_stats")
                     )
                     logging.debug(f"Статистика модели для эпохи {current_eval_epoch} сохранена.")
             except Exception as e:
                 logging.warning(f"Не удалось залогировать статистику модели: {e}", exc_info=True)
             
    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)

                self.current_epoch = int(state.epoch) # state.epoch содержит номер завершенной эпохи
        logging.info(f"Завершена эпоха {self.current_epoch}")
        
        #trainer = kwargs.get("trainer", None)
        trainer = getattr(self, "trainer", None)  # Получаем Trainer из callback handler
        # Добавляем проверку state.is_world_process_zero, чтобы избежать дублирования
        if trainer is None or not hasattr(trainer, "model") or not state.is_world_process_zero:
            return

        model = trainer.model

                try:
            if hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
                eval_dataloader = trainer.get_eval_dataloader(trainer.eval_dataset)
                device = trainer.args.device

                all_norms = []
                all_depths = []
                model.eval() # Убедимся, что модель в eval режиме
                with torch.no_grad():
                    for step, inputs in enumerate(eval_dataloader):
                        # Ограничим количество батчей для ускорения
                        if step >= 10: # Обрабатываем только первые 10 батчей
                            break
                        # Корректно извлекаем пути
                        file_paths_batch = inputs.pop("question_file_paths", [])
                        inputs = trainer._prepare_inputs(inputs)

                        # Получаем эмбеддинги вопроса
                        question_encoder_outputs = model.rag.question_encoder(
                            input_ids=inputs.get("input_ids"),
                            attention_mask=inputs.get("attention_mask"),
                            return_dict=True
                        )
                        # Извлекаем эмбеддинги вопроса: используем pooler_output, иначе скрытые состояния
                        if hasattr(question_encoder_outputs, "pooler_output") and question_encoder_outputs.pooler_output is not None:
                            question_embeddings = question_encoder_outputs.pooler_output
                        elif hasattr(question_encoder_outputs, "last_hidden_state"):
                            question_embeddings = question_encoder_outputs.last_hidden_state[:, 0, :]
                        elif hasattr(question_encoder_outputs, "hidden_states") and question_encoder_outputs.hidden_states is not None:
                            question_embeddings = question_encoder_outputs.hidden_states[-1][:, 0, :]
                        else:
                            raise ValueError("Не удалось извлечь эмбеддинги вопроса для Spearman-регуляризации")

                        question_norms = torch.norm(question_embeddings, p=2, dim=1)

                        # Вычисляем глубины
                        for i, paths in enumerate(file_paths_batch):
                            if paths:
                                depths = [len(path.strip('/').split('/')) for path in paths]
                                avg_depth = sum(depths) / len(depths) if depths else 0.0
                                if avg_depth > 0:
                                    all_norms.append(question_norms[i].item())
                                    all_depths.append(avg_depth)

                # Вычисляем и логируем корреляцию, если достаточно данных
                if len(all_norms) > 1:
                    norms_np = np.array(all_norms)
                    depths_np = np.array(all_depths)
                    # Используем scipy.stats.spearmanr, убедившись, что он импортирован
                    from scipy.stats import spearmanr
                    correlation, p_value = spearmanr(norms_np, depths_np)

                    # Проверяем на NaN перед логированием
                    if not np.isnan(correlation):
                        logging.info(f"Корреляция Спирмена (валид. выборка, эпоха {self.current_epoch}): {correlation:.4f} (p={p_value:.4f})")

                        # Вызываем функцию логирования из viz_utils
                        if self.viz_utils_available:
                            try:
                                viz_utils.log_spearman_correlation(
                                    epoch=self.current_epoch,
                                    rho=correlation,
                                    p_value=p_value,
                                    log_file=os.path.join(self.logs_dir, "spearman.csv")
                                )
                            except Exception as e:
                                logging.warning(f"Ошибка при логировании корреляции Спирмена: {e}")
                    else:
                         logging.warning(f"Корреляция Спирмена NaN на эпохе {self.current_epoch}, логирование пропущено.")

        except Exception as e:
            logging.warning(f"Ошибка при вычислении/логировании корреляции Спирмена: {e}", exc_info=True)

                try:
            logging.info(f"Эпоха {self.current_epoch}: Проверка условия для логирования центроидов.")
            logging.info(f"Эпоха {self.current_epoch}: self.viz_utils_available = {self.viz_utils_available}")
            logging.info(f"Эпоха {self.current_epoch}: self.current_epoch % self.log_centroids_every = {self.current_epoch % self.log_centroids_every}")
            if self.viz_utils_available:  # Логируем центроиды каждую эпоху
                 logging.info(f"Эпоха {self.current_epoch}: Условие для логирования центроидов выполнено.")
                 if hasattr(model, "rag") and hasattr(model.rag, "retriever") and hasattr(model.rag.retriever, "structural_embeddings"):
                     logging.info(f"Эпоха {self.current_epoch}: Найдены structural_embeddings в модели.")
                     # Получаем подвыборку индексов из eval_dataset
                     sample_size = min(1000, len(trainer.eval_dataset))
                     indices = np.random.choice(len(trainer.eval_dataset), size=sample_size, replace=False)
                     logging.info(f"Эпоха {self.current_epoch}: Выбрано {sample_size} случайных индексов для семплирования.")

                     # Получаем все структурные эмбеддинги из ретривера
                     all_struct_embeddings = model.rag.retriever.structural_embeddings.detach().cpu()
                     logging.info(f"Эпоха {self.current_epoch}: Размер all_struct_embeddings: {all_struct_embeddings.shape}")

                     # Выбираем эмбеддинги по индексам
                     if len(all_struct_embeddings) >= np.max(indices) + 1: # Исправлено условие
                          sampled_embeddings = all_struct_embeddings[indices]
                          logging.info(f"Эпоха {self.current_epoch}: Размер sampled_embeddings: {sampled_embeddings.shape}")
                     else:
                          # Если индексы выходят за пределы, берем доступные
                          valid_indices = indices[indices < len(all_struct_embeddings)]
                          sampled_embeddings = all_struct_embeddings[valid_indices]
                          logging.warning(f"Эпоха {self.current_epoch}: Не все индексы ({len(indices)}) были валидны для логирования центроидов (размер KB: {len(all_struct_embeddings)}). Залогировано: {len(sampled_embeddings)}.")

                     # Логируем центроиды для текущей эпохи
                     if sampled_embeddings.numel() > 0: # Проверяем, что есть что логировать
                         logging.info(f"Эпоха {self.current_epoch}: sampled_embeddings не пустой ({sampled_embeddings.numel()} элементов). Вызов viz_utils.log_centroid_epoch.")
                         viz_utils.log_centroid_epoch(
                             epoch=self.current_epoch,
                             centroids=sampled_embeddings.numpy(), # Передаем numpy массив
                             log_dir=self.logs_dir
                         )
                     else:
                         logging.warning(f"Эпоха {self.current_epoch}: sampled_embeddings пустой. Логирование центроидов пропущено.")
                 else:
                     logging.warning(f"Эпоха {self.current_epoch}: model.rag.retriever.structural_embeddings не найдены.")
            else:
                logging.info(f"Эпоха {self.current_epoch}: Условие для логирования центроидов НЕ выполнено.")
        except Exception as e:
            logging.warning(f"Ошибка при логировании центроидов: {e}", exc_info=True)

        # Добавляю запись инференса в CSV
        try:
            val_data = load_split_data(CONFIG, "validation")
            if val_data is not None:
                diffs = val_data['diffs']
                orig_msgs = val_data['messages']
                paths = val_data.get('file_paths', [])
                n = len(diffs)
                sample_n = 20 if n >= 20 else n
                indices_inf = np.random.choice(n, size=sample_n, replace=False)
                results = []
                for idx in indices_inf:
                    diff_text = diffs[idx]
                    orig_msg = orig_msgs[idx]
                    fp = paths[idx] if idx < len(paths) else []
                    encoded = trainer.tokenizer.question_encoder(
                        diff_text,
                        return_tensors="pt",
                        max_length=CONFIG.get("question_encoder_max_length", 512),
                        padding="max_length",
                        truncation=True
                    )
                    input_ids = encoded.input_ids.to(trainer.args.device)
                    attention_mask = encoded.attention_mask.to(trainer.args.device)
                    with torch.no_grad():
                        gen_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            question_file_paths=[fp],
                            generation_config=model.generation_config
                        )
                    gen_msg = trainer.tokenizer.generator.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    results.append({
                        "epoch": self.current_epoch,
                        "diff": diff_text,
                        "original_message": orig_msg,
                        "generated_message": gen_msg
                    })
                df_inf = pd.DataFrame(results)
                csv_path = os.path.join(self.logs_dir, f"inference_validation_epoch{self.current_epoch}.csv")
                df_inf.to_csv(csv_path, index=False, encoding='utf-8')
                logging.info(f"Сохранено {len(results)} примеров инференса для валидации эпохи{self.current_epoch} в {csv_path}")
        except Exception as e:
            logging.warning(f"Ошибка при записи примеров инференса в CSV: {e}", exc_info=True)
        # Конец добавления

    def on_train_end(self, args, state, control, **kwargs):
        """Вызывается в конце обучения для финальной визуализации."""
        super().on_train_end(args, state, control, **kwargs)

        # Добавляем проверку state.is_world_process_zero
        if not self.viz_utils_available or not state.is_world_process_zero:
            return

        logging.info("Завершение обучения. Генерация финальных визуализаций...")

                try:
            centroid_files = glob.glob(os.path.join(self.logs_dir, "centroids_epoch*.npy"))
            if len(centroid_files) >= 2:
                viz_utils.create_centroid_drift_animation(
                    centroids_pattern=os.path.join(self.logs_dir, "centroids_epoch*.npy"),
                    output_path=os.path.join(self.figures_dir, "centroid_drift_final.gif"),
                    fps=2
                )
                logging.info(f"Финальная анимация дрейфа центроидов создана в {self.figures_dir}")
        except Exception as e:
            logging.warning(f"Ошибка при создании финальной анимации: {e}")

                try:
            spearman_log = os.path.join(self.logs_dir, "spearman.csv")
            if os.path.exists(spearman_log):
                viz_utils.plot_spearman_correlation(
                    log_file=spearman_log,
                    output_path=os.path.join(self.figures_dir, "spearman_curve_final.png"),
                    figsize=(8, 5),
                    dpi=300 # Увеличим разрешение для финального графика
                )
                logging.info(f"Финальный график корреляции Спирмена создан в {self.figures_dir}")
        except Exception as e:
            logging.warning(f"Ошибка при создании финального графика корреляции: {e}")

                try:
            topk_log = os.path.join(self.logs_dir, "topk.csv")
            if os.path.exists(topk_log):
                 viz_utils.plot_topk_accuracy(
                     data_file=topk_log,
                     output_path=os.path.join(self.figures_dir, "topk_bar_final.png"),
                     figsize=(6, 4),
                     dpi=300 # Увеличим разрешение
                 )
                 logging.info(f"Финальный график Top-k accuracy создан в {self.figures_dir}")
            else:
                 logging.warning(f"Файл {topk_log} не найден. График Top-k accuracy не будет создан.")
        except Exception as e:
            logging.warning(f"Ошибка при создании финального графика Top-k accuracy: {e}")

                try:
            mixing_log = os.path.join(self.logs_dir, "mixing_scores.csv")
            if os.path.exists(mixing_log):
                 viz_utils.plot_mixing_heatmap(
                     data_file=mixing_log,
                     output_path=os.path.join(self.figures_dir, "mixing_heatmap_final.png"),
                     figsize=(6, 5),
                     dpi=300 # Увеличим разрешение
                 )
                 logging.info(f"Финальная тепловая карта смешивания создана в {self.figures_dir}")
            else:
                 logging.warning(f"Файл {mixing_log} не найден. Тепловая карта смешивания не будет создана.")
        except Exception as e:
            logging.warning(f"Ошибка при создании финальной тепловой карты смешивания: {e}")

                try:
            model = kwargs.get("model", None)
            if model is not None:
                final_epoch = int(state.epoch) if state.epoch is not None else self.current_epoch
                if final_epoch is None: # Доп. проверка на случай если state.epoch все еще None
                   final_epoch = self.current_epoch
                viz_utils.save_model_stats(
                    model=model,
                    epoch=final_epoch if final_epoch is not None else -1, # Используем -1 если эпоха неизвестна
                    output_dir=os.path.join(self.logs_dir, "model_stats")
                )
                logging.info(f"Финальная статистика модели сохранена для эпохи {final_epoch}.")
        except Exception as e:
            logging.warning(f"Не удалось сохранить финальную статистику модели: {e}", exc_info=True)


def spearmanr(a, b, axis=0):
    """
    Вычисляет корреляцию Спирмена между двумя наборами данных.
    
    Args:
        a: Первый набор данных (тензор или массив)
        b: Второй набор данных (тензор или массив)
        axis: Ось, по которой вычисляется корреляция
        
    Returns:
        Коэффициент корреляции Спирмена
    """
    # Проверяем размеры входных данных
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    
    # Проверяем, что массивы имеют одинаковую форму
    if a.shape != b.shape:
        raise ValueError(f"Формы a и b должны совпадать: {a.shape} != {b.shape}")
    
    # Вычисляем ранги
    def _rankdata(x):
        """
        Ранжирует элементы массива.
        """
        # Сортируем элементы и получаем индексы сортировки
        sorter = np.argsort(x, axis=axis)
        
        # Инициализируем массив рангов
        ranks = np.zeros_like(x, dtype=np.float64)
        
        # Заполняем ранги
        for i in range(x.shape[axis]):
            ranks[sorter[i]] = i + 1
        
        # Обрабатываем связи (одинаковые значения)
        # Находим дубликаты и заменяем их ранги средним
        unique_values, value_counts = np.unique(x, return_counts=True)
        for value, count in zip(unique_values, value_counts):
            if count > 1:
                # Находим индексы элементов с данным значением
                value_indices = np.where(x == value)[0]
                # Вычисляем средний ранг
                mean_rank = np.mean(ranks[value_indices])
                # Заменяем ранги средним значением
                ranks[value_indices] = mean_rank
        
        return ranks
    
    # Ранжируем данные
    a_ranks = _rankdata(a)
    b_ranks = _rankdata(b)
    
    # Вычисляем корреляцию Спирмена
    n = a.shape[0]
    rs_num = np.sum((a_ranks - np.mean(a_ranks)) * (b_ranks - np.mean(b_ranks)))
    rs_den = np.sqrt(np.sum((a_ranks - np.mean(a_ranks))**2) * np.sum((b_ranks - np.mean(b_ranks))**2))
    
    if rs_den == 0:
        return 0.0  # Если знаменатель равен нулю, возвращаем 0
    
    rs = rs_num / rs_den
    
    return rs


def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Вычисляет потерю с добавлением регуляризации на основе корреляции Спирмена 
    между нормами эмбеддингов и глубиной путей.
    """
    # Извлекаем file_paths
    question_file_paths = inputs.pop("question_file_paths", None)
    
    # Вычисляем стандартную потерю модели
    outputs = model(**inputs, question_file_paths=question_file_paths, return_dict=True)
    loss = outputs.loss
    
    # Если loss это тензор с размером > 0, берем его среднее
    if loss is not None and loss.dim() > 0:
        loss = loss.mean()
    
    # Добавляем регуляризацию на основе корреляции Спирмена, если указаны пути файлов
    spearman_weight = CONFIG.get("spearman_weight", 0.1)
    
    if spearman_weight > 0 and question_file_paths is not None and hasattr(model, "get_input_embeddings"):
        try:
            # Получаем нормы вопросов из эмбеддингов модели
            question_embeddings = outputs.get("question_embeddings")
            if question_embeddings is None and hasattr(outputs, "encoder_outputs"):
                # Для моделей RAG, где вопрос закодирован в encoder_outputs
                question_embeddings = outputs.encoder_outputs[0][:, 0, :]  # Берем CLS-токен
            
            if question_embeddings is not None:
                # Вычисляем нормы эмбеддингов
                question_norms = torch.norm(question_embeddings, dim=1)
                
                # Вычисляем средние глубины путей для каждого вопроса
                depth_values = []
                for paths in question_file_paths:
                    if paths:
                        # Вычисляем глубину как среднее количество компонентов в пути
                        depths = [len(path.strip('/').split('/')) for path in paths]
                        avg_depth = sum(depths) / len(depths)
                        depth_values.append(avg_depth)
                    else:
                        depth_values.append(0.0)  # Для случаев без путей
                
                # Преобразуем в тензор
                path_depths = torch.tensor(depth_values, device=question_norms.device)
                
                # Вычисляем корреляцию Спирмена, только если есть валидные глубины
                if torch.sum(path_depths > 0) > 1:
                    spearman_corr = spearmanr(question_norms.detach().cpu(), path_depths.detach().cpu())
                    
                    # Целевая корреляция должна быть положительной (больше нормы для более глубоких путей)
                    spearman_loss = -spearman_corr  # Инвертируем, чтобы минимизировать потерю
                    # Добавляем к общей потере
                    loss = loss + spearman_weight * spearman_loss
        except Exception as e:
            # Логируем ошибку, но не останавливаем обучение
            print(f"Ошибка при вычислении регуляризации Спирмена: {e}")
    
    # Возвращаем результат
    return (loss, outputs) if return_outputs else loss


def main():
    
        log_dir = CONFIG.get("logging_dir", os.path.join(CONFIG["prepared_data_dir"], 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "training.log")
    
        log_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root_logger = logging.getLogger() # Получаем корневой логгер
    root_logger.setLevel(logging.INFO) # Устанавливаем уровень
    root_logger.handlers.clear() # Очищаем существующие обработчики (на всякий случай)
    
    # Добавляем обработчик для вывода в консоль
        stream_handler = logging.StreamHandler()
    stream_handler.encoding = 'utf-8' # Устанавливаем кодировку для консоли
        stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)
    
    # Добавляем обработчик для вывода в файл
    try:
                file_handler = logging.FileHandler(log_file_path, encoding='utf-8') # Устанавливаем кодировку для файла
                file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        print(f"Логирование настроено. Файл логов: {log_file_path}")
    except Exception as e:
        print(f"Ошибка настройки файлового логгера: {e}")
        
    # Устанавливаем уровень логера transformers (чтобы видеть его сообщения)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.DEBUG) # Устанавливаем DEBUG и для него
    transformers_logger.propagate = True
    
            if not any(isinstance(h, logging.StreamHandler) for h in transformers_logger.handlers):
        transformers_logger.addHandler(stream_handler)
    if not any(isinstance(h, logging.FileHandler) for h in transformers_logger.handlers):
        try:
            transformers_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Не удалось добавить file_handler к логгеру transformers: {e}")
        
        # Чтобы скрыть INFO сообщения о времени поиска в FAISS
    logging.getLogger("transformers.models.rag.retrieval_rag").setLevel(logging.WARNING)
        
        figures_dir = os.path.join(CONFIG["prepared_data_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    if _viz_utils_available:
        logging.info(f"Модуль визуализации доступен. Графики будут сохраняться в {figures_dir}")
    else:
        logging.warning("Модуль визуализации недоступен. Графики и анимации не будут создаваться.")
        
    # 0. Подготовка ВСЕХ данных
    print("--- Шаг 0: Подготовка данных ---")
    data_ready = prepare_all_data(CONFIG)
    
    if data_ready:
        # 1. Инициализация модели и токенизатора
        print("\n--- Шаг 1: Инициализация модели --- ")
        tokenizer, model = initialize_reranking_rag_model(config=CONFIG)
        
        if model and tokenizer:
            print("\nМодель и токенизатор успешно созданы.")

            # 2. Подготовка данных для обучения
            print("\n--- Шаг 2: Подготовка датасетов для обучения ---")
            # ПРЕДПОЛАГАЕТСЯ, что train_data.pkl и validation_data.pkl существуют!
            train_data_dict = load_split_data(CONFIG, "train")
            val_data_dict = load_split_data(CONFIG, "validation")
            
            if train_data_dict and val_data_dict:
                tokenized_train_dataset = tokenize_dataset_for_training(
                    train_data_dict, tokenizer, CONFIG)
                tokenized_val_dataset = tokenize_dataset_for_training(
                    val_data_dict, tokenizer, CONFIG)
                
                if tokenized_train_dataset is None or tokenized_val_dataset is None:
                    print("Ошибка: Не удалось токенизировать данные. Обучение прервано.")
                    exit()
                    
                print(f"Тренировочный датасет создан, размер: {len(tokenized_train_dataset)}")
                print(f"Валидационный датасет создан, размер: {len(tokenized_val_dataset)}")

                                val_length = len(tokenized_val_dataset)
                max_val_samples = 7500
                val_samples_to_use = min(val_length, max_val_samples)
                print(f"Используем только {val_samples_to_use} примеров из {val_length} для валидации (ограничено {max_val_samples})")
                # Создаем подмножество валидационного набора с первыми half_val_length примерами
                tokenized_val_dataset = tokenized_val_dataset.select(range(val_samples_to_use))
                
                # 3. Настройка Data Collator
                data_collator = CustomDataCollatorWithPaths(
                    tokenizer=tokenizer.generator, 
                    model=model, 
                    label_pad_token_id=model.config.pad_token_id, 
                    padding='longest'
                )
                print("Кастомный Data Collator настроен.")
                
                # 4. Настройка Training Arguments (Используем Seq2SeqTrainingArguments)
                                output_dir = os.path.join(CONFIG["prepared_data_dir"], CONFIG.get("output_dir_base", "training_output"))
                training_args = Seq2SeqTrainingArguments(
                    output_dir=output_dir,
                    overwrite_output_dir=True,
                    num_train_epochs=CONFIG["epochs"],
                    per_device_train_batch_size=CONFIG["batch_size"],
                    gradient_accumulation_steps=CONFIG.get("gradient_accumulation_steps", 1),
                    learning_rate=CONFIG["learning_rate"],
                    # Используем явную настройку из конфигурации, чтобы избежать ошибки AMP на CPU
                    fp16=CONFIG.get("use_fp16", False),
                    save_strategy=CONFIG.get("save_strategy", "epoch"),
                    evaluation_strategy="epoch",
                    load_best_model_at_end=CONFIG.get("load_best_model_at_end", False),
                    metric_for_best_model=CONFIG.get("metric_for_best_model", "eval_loss"),
                    greater_is_better=CONFIG.get("greater_is_better", False),
                    save_total_limit=CONFIG.get("save_total_limit", None),
                    warmup_steps=CONFIG.get("warmup_steps", 0),
                    weight_decay=CONFIG.get("weight_decay", 0.0),
                    logging_dir=os.path.join(output_dir, 'runs'),
                    logging_strategy="steps",
                    logging_steps=CONFIG.get("logging_steps", 50),
                    logging_first_step=True,
                    report_to=CONFIG.get("report_to", "none"),
                    remove_unused_columns=False,
                    predict_with_generate=True,
                    generation_max_length=CONFIG.get("generation_max_length", 128),
                    generation_num_beams=CONFIG.get("generation_num_beams", 1)
                )
                                print("Seq2SeqTraining Arguments настроены.")
                
                                if _viz_utils_available:
                    try:
                        # Загружаем структурные эмбеддинги 
                        struct_emb_path = CONFIG["structural_embeddings_output_path"]
                        if os.path.exists(struct_emb_path):
                            logging.info("Отрисовка начального диска Пуанкаре...")
                            
                            struct_embeddings = torch.load(struct_emb_path)
                            
                            # Выбираем подмножество для отрисовки
                            sample_size = min(1000, len(struct_embeddings))
                            indices = np.random.choice(len(struct_embeddings), size=sample_size, replace=False)
                            
                            sampled_centroids = struct_embeddings[indices]
                            
                            # Отрисовываем диск Пуанкаре
                            viz_utils.plot_poincare_disk(
                                centroids=sampled_centroids,
                                output_path=os.path.join(figures_dir, "initial_poincare_disk.png"),
                                figsize=(8, 8),
                                dpi=300
                            )
                            logging.info(f"Начальный диск Пуанкаре отрисован")
                    except Exception as e:
                        logging.warning(f"Ошибка при отрисовке начального диска Пуанкаре: {e}")
                                
                # 5. Создание и запуск Тренера (Используем RerankingSeq2SeqTrainer)
                                callbacks = []
                
                # Добавляем наш LoggingCallback для визуализации и логирования
                logging_callback = LoggingCallback(early_stopping_patience=CONFIG.get("early_stopping_patience", 3))
                callbacks.append(logging_callback)
                
                # Если включен tensorboard, добавляем его callback
                if CONFIG.get("report_to", "none") == "tensorboard":
                    from transformers.integrations import TensorBoardCallback
                    callbacks.append(TensorBoardCallback())
                
                trainer = RerankingSeq2SeqTrainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=tokenized_train_dataset,
                    eval_dataset=tokenized_val_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer.generator),
                    callbacks=callbacks
                )
                # Передаем ссылку на Trainer в колбэках для логирования центроидов и инференса
                for callback in callbacks:
                    callback.set_trainer(trainer)
                print("\n--- Шаг 3: Запуск обучения --- ")
                try:
                    train_result = trainer.train()
                    print("--- Обучение завершено --- ")
                    print(train_result)
                    
                                        if _viz_utils_available:
                        try:
                            # Сбор финальной статистики модели
                            viz_utils.save_model_stats(
                                model=model,
                                epoch=999,  # Специальное значение для финальной статистики
                                output_dir=os.path.join(log_dir, "model_stats")
                            )
                            
                            # Создаем финальные графики с большим размером и разрешением
                            # Топ-k точность
                            viz_utils.plot_topk_accuracy(
                                data_file=os.path.join(log_dir, "topk.csv"),
                                output_path=os.path.join(figures_dir, "topk_bar_final.png"),
                                figsize=(8, 6),
                                dpi=400
                            )
                            
                            # Тепловая карта смешивания
                            if os.path.exists(os.path.join(log_dir, "mixing_scores.csv")):
                                viz_utils.plot_mixing_heatmap(
                                    data_file=os.path.join(log_dir, "mixing_scores.csv"),
                                    output_path=os.path.join(figures_dir, "mixing_heatmap_final.png"),
                                    figsize=(8, 6),
                                    dpi=400
                                )
                            
                            logging.info("Финальные визуализации созданы")
                        except Exception as e:
                            logging.warning(f"Ошибка при создании финальных визуализаций: {e}")
                                        
                except Exception as e:
                    print(f"\nОшибка во время обучения: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                print("\nНе удалось загрузить данные для обучения/валидации.")
        else:
            print("\nНе удалось инициализировать модель для обучения.")
    else:
        print("\nПодготовка данных не удалась. Обучение отменено.")
if __name__ == "__main__":
    main()

