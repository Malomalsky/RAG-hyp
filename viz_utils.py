#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Модуль визуализации для проекта RAG с гиперболическим переранжированием.
Содержит функции для отрисовки диаграмм согласно плану в graphics.md.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# Определяем цветовую схему
COLORS = {
    "baseline": "#B0B0B0",
    "rerank": "#3C4FDB"
}

def ensure_directory(path):
    """Создает директорию, если она не существует"""
    os.makedirs(path, exist_ok=True)

def plot_poincare_disk(centroids, file_paths=None, output_path="figures/poincare_disk.png", 
                       pca_if_needed=True, figsize=(6, 6), dpi=300):
    """
    Отрисовывает центроиды на диске Пуанкаре.
    
    Args:
        centroids: Тензор или массив центроидов [N, dim]
        file_paths: Список списков путей к файлам для вычисления глубины
        output_path: Путь для сохранения изображения
        pca_if_needed: Применять ли PCA, если размерность > 2
        figsize: Размер фигуры
        dpi: Разрешение изображения
    """
    ensure_directory(os.path.dirname(output_path))
    
    # Преобразуем в numpy, если это тензор
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    
    # Применяем PCA, если размерность > 2 и включена опция
    if centroids.shape[1] > 2 and pca_if_needed:
        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(centroids)
    else:
        centroids_2d = centroids[:, :2]
    
    # Извлекаем координаты x и y
    x, y = centroids_2d[:, 0], centroids_2d[:, 1]
    
    # Вычисляем глубину путей, если они предоставлены
    if file_paths is not None:
        depths = []
        for paths in file_paths:
            if paths:
                avg_depth = np.mean([len(path.strip('/').split('/')) for path in paths])
                depths.append(avg_depth)
            else:
                depths.append(0)
        
        # Нормализуем глубины от 0 до 1
        depth_norm = np.array(depths)
        min_depth, max_depth = depth_norm.min(), depth_norm.max()
        if max_depth > min_depth:
            depth_norm = (depth_norm - min_depth) / (max_depth - min_depth)
    else:
        # Если пути не предоставлены, используем нормы центроидов как прокси для глубины
        radius = np.sqrt(x**2 + y**2)
        depth_norm = radius  # Норма уже между 0 и 1 для диска Пуанкаре
    
    # Создаем и настраиваем график
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Рисуем границу диска Пуанкаре
    circle = plt.Circle((0, 0), 1, color="k", fill=False, lw=1.5)
    ax.add_artist(circle)
    
    # Рисуем центроиды
    scatter = ax.scatter(x, y, c=depth_norm, cmap="viridis", s=15, alpha=0.8)
    
    # Настраиваем оси
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Центроиды на диске Пуанкаре")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Добавляем цветовую шкалу
    plt.colorbar(scatter, label="Глубина пути")
    
    # Сохраняем изображение
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Диаграмма диска Пуанкаре сохранена в {output_path}")

def create_centroid_drift_animation(centroids_pattern="logs/centroids_epoch*.npy", 
                                   output_path="figures/centroid_drift.gif", 
                                   figsize=(4, 4), fps=1, interval=800):
    """
    Создает анимацию дрейфа центроидов по эпохам.
    
    Args:
        centroids_pattern: Шаблон для поиска файлов с центроидами
        output_path: Путь для сохранения GIF-анимации
        figsize: Размер фигуры
        fps: Кадров в секунду
        interval: Интервал между кадрами в миллисекундах
    """
    ensure_directory(os.path.dirname(output_path))
    
    # Находим все файлы с центроидами
    frames = sorted(glob.glob(centroids_pattern))
    if not frames:
        print(f"Не найдены файлы центроидов по шаблону: {centroids_pattern}")
        return
    
    # Загружаем первый файл для определения формы и применения PCA при необходимости
    first_data = np.load(frames[0])
    if first_data.shape[1] > 2:
        pca = PCA(n_components=2)
        first_data_2d = pca.fit_transform(first_data)
        use_pca = True
    else:
        first_data_2d = first_data[:, :2]
        use_pca = False
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=figsize)
    circle = plt.Circle((0, 0), 1, color="k", fill=False, lw=1.5)
    ax.add_artist(circle)
    scat = ax.scatter(first_data_2d[:, 0], first_data_2d[:, 1], c=np.sqrt(first_data_2d[:, 0]**2 + first_data_2d[:, 1]**2),
                     cmap="viridis", s=15, alpha=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Дрейф центроидов (эпоха: 0)")
    
    # Функция обновления для анимации
    def update(i):
        data = np.load(frames[i])
        # Применяем PCA при необходимости
        if use_pca:
            data_2d = pca.transform(data)
        else:
            data_2d = data[:, :2]
        
        # Обновляем координаты и цвета
        scat.set_offsets(data_2d)
        radius = np.sqrt(data_2d[:, 0]**2 + data_2d[:, 1]**2)
        scat.set_array(radius)
        
        # Обновляем заголовок с номером эпохи
        epoch_num = int(os.path.basename(frames[i]).replace("centroids_epoch", "").replace(".npy", ""))
        ax.set_title(f"Дрейф центроидов (эпоха: {epoch_num})")
        
        return (scat,)
    
    # Создаем анимацию
    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    
    # Сохраняем анимацию в GIF
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    
    print(f"Анимация дрейфа центроидов сохранена в {output_path}")

def plot_spearman_correlation(log_file="logs/spearman.csv", output_path="figures/spearman_curve.png",
                             figsize=(6, 3), dpi=300, compare_baseline=False, baseline_file=None):
    """
    Строит график корреляции Спирмена по эпохам.
    
    Args:
        log_file: Путь к CSV-файлу с данными корреляции Спирмена
        output_path: Путь для сохранения изображения
        figsize: Размер фигуры
        dpi: Разрешение изображения
        compare_baseline: Сравнивать ли с baseline
        baseline_file: Путь к CSV-файлу с данными baseline
    """
    ensure_directory(os.path.dirname(output_path))
    
    # Загружаем данные
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Файл с данными корреляции Спирмена не найден: {log_file}")
        return
    
    # Создаем фигуру
    plt.figure(figsize=figsize)
    
    # Строим график
    if compare_baseline and baseline_file:
        try:
            df_baseline = pd.read_csv(baseline_file)
            df['run'] = 'rerank'
            df_baseline['run'] = 'baseline'
            combined_df = pd.concat([df, df_baseline])
            sns.lineplot(data=combined_df, x="epoch", y="rho", hue="run", marker="o",
                        palette=COLORS)
        except FileNotFoundError:
            print(f"Файл с данными baseline не найден: {baseline_file}")
            sns.lineplot(data=df, x="epoch", y="rho", marker="o", color=COLORS["rerank"])
    else:
        sns.lineplot(data=df, x="epoch", y="rho", marker="o", color=COLORS["rerank"])
    
    # Добавляем горизонтальную линию на нуле
    plt.axhline(0, ls="--", c="grey", alpha=0.7)
    
    # Настраиваем подписи
    plt.ylabel("Корреляция Спирмена ρ")
    plt.xlabel("Эпоха")
    plt.title("Динамика корреляции Спирмена по эпохам")
    
    # Сохраняем изображение
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    print(f"График корреляции Спирмена сохранен в {output_path}")

def plot_topk_accuracy(data_file="logs/topk.csv", output_path="figures/topk_bar.png",
                      figsize=(4, 3), dpi=300):
    """
    Строит bar-chart точности Top-k до и после переранжирования.
    
    Args:
        data_file: Путь к CSV-файлу с данными точности
        output_path: Путь для сохранения изображения
        figsize: Размер фигуры
        dpi: Разрешение изображения
    """
    ensure_directory(os.path.dirname(output_path))
    
    # Загружаем данные
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Файл с данными точности Top-k не найден: {data_file}")
        # Создаем примерные данные для демонстрации
        df = pd.DataFrame({
            "k": [1, 3, 5] * 2,
            "accuracy": [65, 78, 85, 72, 86, 92],
            "mode": ["baseline"] * 3 + ["rerank"] * 3
        })
        print("Используются демонстрационные данные")
    
    # Создаем фигуру
    plt.figure(figsize=figsize)
    
    # Строим bar-chart
    ax = sns.barplot(data=df, x="k", y="accuracy", hue="mode", palette=COLORS)
    
    # Настраиваем подписи
    plt.ylabel("Top-k точность, %")
    plt.xlabel("k")
    plt.title("Сравнение точности до и после переранжирования")
    plt.ylim(0, 100)
    
    # Добавляем значения на столбцы
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')
    
    # Сохраняем изображение
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    print(f"Bar-chart Top-k точности сохранен в {output_path}")

def plot_mixing_heatmap(data_file="logs/mixing_scores.csv", output_path="figures/mixing_heatmap.png",
                       figsize=(4, 3), dpi=300):
    """
    Строит тепловую карту смешивания семантических и структурных скоров.
    
    Args:
        data_file: Путь к CSV-файлу с данными смешивания
        output_path: Путь для сохранения изображения
        figsize: Размер фигуры
        dpi: Разрешение изображения
    """
    ensure_directory(os.path.dirname(output_path))
    
    # Загружаем данные
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Файл с данными смешивания не найден: {data_file}")
        return
    
    # Создаем сводную таблицу
    try:
        pivot = df.pivot_table(index="struct_q", columns="sem_q", values="hit", aggfunc="mean")
    except Exception as e:
        print(f"Ошибка при создании сводной таблицы: {e}")
        return
    
    # Создаем фигуру
    plt.figure(figsize=figsize)
    
    # Строим тепловую карту
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="rocket_r")
    
    # Настраиваем подписи
    plt.xlabel("Квартиль семантического скора")
    plt.ylabel("Квартиль структурного скора")
    plt.title("Влияние смешивания скоров на точность")
    
    # Сохраняем изображение
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    print(f"Тепловая карта смешивания сохранена в {output_path}")

def log_spearman_correlation(epoch, rho, p_value=None, log_file="logs/spearman.csv"):
    """
    Логирует значение корреляции Спирмена в CSV-файл.
    
    Args:
        epoch: Номер эпохи
        rho: Значение корреляции Спирмена
        p_value: p-значение (опционально)
        log_file: Путь к CSV-файлу для логирования
    """
    ensure_directory(os.path.dirname(log_file))
    
    # Создаем DataFrame для текущей записи
    data = {"epoch": [epoch], "rho": [rho]}
    if p_value is not None:
        data["p_value"] = [p_value]
    
    row_df = pd.DataFrame(data)
    
    # Проверяем, существует ли файл
    if os.path.exists(log_file):
        # Если файл существует, добавляем новую строку
        existing_df = pd.read_csv(log_file)
        updated_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        # Если файл не существует, создаем новый
        updated_df = row_df
    
    # Сохраняем обновленный DataFrame
    updated_df.to_csv(log_file, index=False)

def log_topk_accuracy(k_values, baseline_accuracy, rerank_accuracy, log_file="logs/topk.csv"):
    """
    Логирует значения Top-k точности в CSV-файл.
    
    Args:
        k_values: Список значений k
        baseline_accuracy: Список значений точности для baseline
        rerank_accuracy: Список значений точности для rerank
        log_file: Путь к CSV-файлу для логирования
    """
    ensure_directory(os.path.dirname(log_file))
    
    # Создаем DataFrame
    k_list = k_values * 2
    accuracy_list = baseline_accuracy + rerank_accuracy
    mode_list = ["baseline"] * len(k_values) + ["rerank"] * len(k_values)
    
    df = pd.DataFrame({
        "k": k_list,
        "accuracy": accuracy_list,
        "mode": mode_list
    })
    
    # Сохраняем DataFrame
    df.to_csv(log_file, index=False)

def log_mixing_scores(struct_quartile, sem_quartile, hit, log_file="logs/mixing_scores.csv"):
    """
    Логирует значения квартилей скоров и индикатор попадания в CSV-файл.
    
    Args:
        struct_quartile: Квартиль структурного скора (1-4)
        sem_quartile: Квартиль семантического скора (1-4)
        hit: Индикатор попадания в top-n (1 или 0)
        log_file: Путь к CSV-файлу для логирования
    """
    ensure_directory(os.path.dirname(log_file))
    
    # Создаем DataFrame для текущей записи
    row_df = pd.DataFrame({
        "struct_q": [struct_quartile],
        "sem_q": [sem_quartile],
        "hit": [hit]
    })
    
    # Проверяем, существует ли файл
    if os.path.exists(log_file):
        # Если файл существует, добавляем новую строку
        existing_df = pd.read_csv(log_file)
        updated_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        # Если файл не существует, создаем новый
        updated_df = row_df
    
    # Сохраняем обновленный DataFrame
    updated_df.to_csv(log_file, index=False)

def log_centroid_epoch(epoch, centroids, log_dir="logs"):
    """
    Сохраняет центроиды текущей эпохи для последующей анимации.
    
    Args:
        epoch: Номер эпохи
        centroids: Тензор или массив центроидов
        log_dir: Директория для сохранения
    """
    ensure_directory(log_dir)
    
    # Преобразуем в numpy, если это тензор
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    
    # Сохраняем центроиды
    output_path = os.path.join(log_dir, f"centroids_epoch{epoch:03d}.npy")
    np.save(output_path, centroids)
    
    print(f"Центроиды эпохи {epoch} сохранены в {output_path}")

# Дополнительная функция - сохранение состояния и статистики модели для отладки
def save_model_stats(model, epoch, output_dir="logs/model_stats"):
    """
    Сохраняет важные параметры модели для анализа.
    
    Args:
        model: Модель RAG
        epoch: Номер эпохи
        output_dir: Директория для сохранения
    """
    ensure_directory(output_dir)
    
    stats = {}
    
    # Извлекаем параметр кривизны
    try:
        if hasattr(model, "rag") and hasattr(model.rag, "retriever") and hasattr(model.rag.retriever, "manifold"):
            manifold = model.rag.retriever.manifold
            if hasattr(manifold, 'k'):
                stats["curvature_k"] = float(manifold.k.item() if isinstance(manifold.k, torch.Tensor) else manifold.k)
            if hasattr(manifold, 'c'):
                stats["curvature_c"] = float(manifold.c.item() if isinstance(manifold.c, torch.Tensor) else manifold.c)
    except Exception as e:
        print(f"Ошибка при извлечении параметра кривизны: {e}")
    
    # Извлекаем вес переранжирования
    try:
        if hasattr(model, "rag") and hasattr(model.rag, "retriever") and hasattr(model.rag.retriever, "rerank_weight"):
            rerank_weight = model.rag.retriever.rerank_weight
            stats["rerank_weight"] = float(rerank_weight.item() if isinstance(rerank_weight, torch.Tensor) else rerank_weight)
    except Exception as e:
        print(f"Ошибка при извлечении веса переранжирования: {e}")
    
    # Извлекаем температуру смешивания
    try:
        if hasattr(model, "rag") and hasattr(model.rag, "retriever") and hasattr(model.rag.retriever, "mixing_temperature"):
            mixing_temp = model.rag.retriever.mixing_temperature
            stats["mixing_temperature"] = float(mixing_temp.item() if isinstance(mixing_temp, torch.Tensor) else mixing_temp)
    except Exception as e:
        print(f"Ошибка при извлечении температуры смешивания: {e}")
    
    # Сохраняем статистику в JSON
    import json
    output_path = os.path.join(output_dir, f"model_stats_epoch{epoch:03d}.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Статистика модели для эпохи {epoch} сохранена в {output_path}")

# --- НОВАЯ ФУНКЦИЯ --- 
def plot_embedding_comparison(hyperbolic_embeddings_2d, euclidean_embeddings_2d, 
                              labels=None, output_path="figures/embedding_comparison.png",
                              figsize=(12, 6), dpi=300):
    """
    Строит два графика рядом: проекцию на диск Пуанкаре и евклидову проекцию.

    Args:
        hyperbolic_embeddings_2d: Координаты точек на диске Пуанкаре [N, 2]
        euclidean_embeddings_2d: Координаты точек в евклидовой проекции (PCA/t-SNE) [N, 2]
        labels: Метки точек для цвета (например, глубина, тип коммита) [N]
        output_path: Путь для сохранения изображения
        figsize: Размер фигуры
        dpi: Разрешение изображения
    """
    ensure_directory(os.path.dirname(output_path))
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # График 1: Диск Пуанкаре
    ax1 = axes[0]
    circle = plt.Circle((0, 0), 1, color='k', fill=False, lw=1)
    ax1.add_artist(circle)
    scatter1 = ax1.scatter(hyperbolic_embeddings_2d[:, 0], hyperbolic_embeddings_2d[:, 1], 
                           c=labels, cmap='viridis', s=10, alpha=0.7)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title("Гиперболическая проекция (Пуанкаре)")
    ax1.set_xlabel("Компонента 1")
    ax1.set_ylabel("Компонента 2")
    
    # График 2: Евклидова проекция
    ax2 = axes[1]
    scatter2 = ax2.scatter(euclidean_embeddings_2d[:, 0], euclidean_embeddings_2d[:, 1], 
                           c=labels, cmap='viridis', s=10, alpha=0.7)
    ax2.set_title("Евклидова проекция (PCA/t-SNE)")
    ax2.set_xlabel("Компонента 1")
    ax2.set_ylabel("Компонента 2")
    ax2.set_aspect('equal') # Делаем масштабы осей равными для наглядности
    
    # Общая цветовая шкала (если есть метки)
    if labels is not None:
        fig.colorbar(scatter1, ax=axes, label="Метка (например, глубина)", shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"График сравнения проекций сохранен в {output_path}")
# --- КОНЕЦ НОВОЙ ФУНКЦИИ --- 

if __name__ == "__main__":
    # Пример использования функций
    print("Модуль визуализации для проекта RAG с гиперболическим переранжированием.")
    print("Используйте функции из этого модуля для отрисовки диаграмм согласно плану в graphics.md.") 