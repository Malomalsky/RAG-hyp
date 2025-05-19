import torch
import geoopt
from typing import Set, List, Optional
import re 
import os 
import math
import hashlib
import logging

DEFAULT_DTYPE = torch.float32

poincare_ball = geoopt.PoincareBall(c=1.0)
try:
    poincare_ball.k.requires_grad_(False)
except AttributeError:

    pass

def project_to_poincare_ball(vectors: torch.Tensor, manifold: geoopt.PoincareBall = poincare_ball, dim: int = -1) -> torch.Tensor:
    """
    Проецирует евклидовы векторы на диск Пуанкаре.

    Args:
        vectors: Входные евклидовы векторы (torch.Tensor).
        manifold: Экземпляр многообразия PoincareBall.
        dim: Измерение, вдоль которого выполняются операции.

    Returns:
        Векторы, спроецированные на диск Пуанкаре (torch.Tensor).
    """
    # Убедимся, что входной тензор имеет нужный dtype
    vectors_typed = vectors.to(DEFAULT_DTYPE)
    projected_vectors = manifold.projx(vectors_typed, dim=dim)
    return projected_vectors

def poincare_distance(vectors1: torch.Tensor, vectors2: torch.Tensor, manifold: geoopt.PoincareBall = poincare_ball, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Вычисляет расстояние Пуанкаре между парами точек на диске Пуанкаре.

    Args:
        vectors1: Первый набор точек на диске Пуанкаре (torch.Tensor).
                 Предполагается, что они уже лежат на диске (результат проекции).
        vectors2: Второй набор точек на диске Пуанкаре (torch.Tensor).
                 Предполагается, что они уже лежат на диске.
        manifold: Экземпляр многообразия PoincareBall.
        dim: Измерение, представляющее признаки многообразия.
        keepdim: Сохранять ли измерение, по которому вычисляется расстояние.

    Returns:
        Расстояния Пуанкаре между соответствующими векторами (torch.Tensor).
    """
    # Убедимся, что входные тензоры имеют нужный dtype
    vec1_typed = vectors1.to(DEFAULT_DTYPE)
    vec2_typed = vectors2.to(DEFAULT_DTYPE)
    
    
    dist = manifold.dist(vec1_typed, vec2_typed, dim=dim, keepdim=keepdim)
    return dist

def extract_file_paths(diff_string: str) -> Set[str]:
    """
    Извлекает уникальные пути измененных файлов из строки diff.
    Фокусируется на новых путях (после 'ppp b /') и исключает '/dev/null'.

    Args:
        diff_string: Строка в формате diff (с <nl> вместо переносов).

    Returns:
        Множество уникальных путей измененных файлов.
    """
    if not isinstance(diff_string, str):
        return set()
    # Ищем строки вида 'ppp b /path/to/file<nl>'
    matches = re.findall(r'^ppp b /(.*?)<nl>', diff_string, re.MULTILINE)
    paths = {match.strip() for match in matches if match.strip() != '/dev/null'}
    return paths

def find_lowest_common_ancestor(paths: Set[str]) -> str:
    """
    Находит самый глубокий общий путь-предок (директорию) для набора путей.

    Args:
        paths: Множество строк-путей.

    Returns:
        Строка, представляющая LCA. Возвращает пустую строку "", если
        общий предок - корень, или если входное множество пустое/содержит только корень.
    """
    if not paths:
        return ""
    # Используем os.path.normpath для унификации разделителей и путей
    # Приводим к строке на всякий случай
    normalized_paths = [os.path.normpath(str(p)).replace("\\", "/") for p in paths]
    # Разделяем на компоненты, убирая пустые строки (например, от начального '/')
    split_paths = [
        [comp for comp in path.split('/') if comp] 
        for path in normalized_paths
    ]
    
    if not split_paths: return ""
    # Если есть пустые списки компонентов (например, путь был просто "/"), LCA - корень
    if any(not p for p in split_paths): return ""

    min_len = min(len(p) for p in split_paths)
    lca_components = []
    for i in range(min_len):
        current_component = split_paths[0][i]
        if all(p[i] == current_component for p in split_paths):
            lca_components.append(current_component)
        else:
            break
    # Собираем обратно с правильным разделителем
    return "/".join(lca_components)

def get_heuristic_hyperbolic_embedding(
    lca_path: str, 
    alpha: float, 
    embedding_dim: int,
    dtype: torch.dtype = DEFAULT_DTYPE, 
    manifold: geoopt.PoincareBall = poincare_ball
) -> torch.Tensor:
    """Вычисляет n-мерный гиперболический эмбеддинг для LCA пути."""
    if not isinstance(lca_path, str): lca_path = ""
    depth = len([comp for comp in lca_path.split('/') if comp])
    
    # 1. Вычисление радиуса
    radius = math.tanh(alpha * depth) 

    # 2. Генерация n-мерного направления на основе хэша
    path_bytes = lca_path.encode('utf-8')
    seed = int.from_bytes(hashlib.sha256(path_bytes).digest()[:8], 'big')
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    direction = torch.randn(embedding_dim, generator=generator, dtype=dtype)
    direction_norm = torch.norm(direction)
    unit_direction = direction / (direction_norm + 1e-9) 
    
    # 3. Получение точки в Евклидовом пространстве
    euclidean_point = radius * unit_direction
    
    # 4. Проекция на диск Пуанкаре
    hyperbolic_embedding = manifold.projx(euclidean_point)

    return hyperbolic_embedding

def calculate_hyperbolic_centroid(points_list: List[torch.Tensor],
                                  embedding_dim: int, 
                                  weights: Optional[List[float]] = None,
                                  manifold: geoopt.PoincareBall = poincare_ball,
                                  max_iterations: int = 50,
                                  lr: float = 0.5, 
                                  eps: float = 1e-4,
                                  clip_threshold: float = 0.999) -> torch.Tensor:
    """Вычисляет гиперболический центроид (среднее Фреше) на диске Пуанкаре."""
    
    if not points_list:
        return torch.zeros(embedding_dim, dtype=DEFAULT_DTYPE)
        
    points = torch.stack(points_list).to(DEFAULT_DTYPE) 
    num_points = points.shape[0]
    actual_dim = points.shape[1]
    if actual_dim != embedding_dim:
        logging.warning(f"В calculate_hyperbolic_centroid переданы точки размерности {actual_dim}, но ожидалась {embedding_dim}. Вычисление будет произведено в размерности {actual_dim}.")
        embedding_dim = actual_dim
    
    # Обработка весов
    if weights is None or len(weights) != num_points:
        if weights is not None:
             logging.warning(f"Длина весов ({len(weights)}) != кол-во точек ({num_points}). Используются единичные веса.")
        weights_tensor = torch.ones(num_points, 1, dtype=DEFAULT_DTYPE, device=points.device)
    else:
        weights_tensor = torch.tensor(weights, dtype=DEFAULT_DTYPE, device=points.device).unsqueeze(1)
        weights_sum = weights_tensor.sum()
        if weights_sum > 1e-6:
            weights_tensor = weights_tensor / weights_sum
        else:
            weights_tensor = torch.ones_like(weights_tensor) / num_points

    # Инициализация и оптимизация
    centroid = torch.mean(points, dim=0)
    centroid = manifold.projx(centroid) 
    centroid.requires_grad_(True)
    optimizer = geoopt.optim.RiemannianAdam([centroid], lr=lr)

    for i in range(max_iterations):
        optimizer.zero_grad()
        dists_sq = manifold.dist(centroid.unsqueeze(0).expand_as(points), points, keepdim=True).pow(2)
        loss = (dists_sq * weights_tensor).sum()
        loss.backward()
        grad_norm = centroid.grad.norm() if centroid.grad is not None else 0
        optimizer.step()
        centroid.data = manifold.projx(centroid.data)
        
        # Клиппинг радиуса
        norm = torch.norm(centroid.data)
        if norm >= clip_threshold:
            centroid.data = centroid.data * (clip_threshold / (norm + 1e-8))
        
        if grad_norm < eps:
            break
            
    return centroid.detach()
