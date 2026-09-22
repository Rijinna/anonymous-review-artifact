import torch
import numpy as np
import scipy.sparse
import tqdm
import unicodedata
import collections

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def norm(s):
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s)).strip().casefold()

def in_ipynb():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def smart_tqdm():  # pragma: no cover
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


# ==========================================
# 核心工具类：OntologyManager
# ==========================================
class OntologyManager:
    """
    scBOL+ 核心工具类：支持层级向心损失计算与语义映射。

    新增属性（相比旧版）：
    - fine_to_coarse_mask : [N, C_coarse] bool tensor
        fine_to_coarse_mask[i, k] = True 表示全局节点 i 属于第 k 个 coarse 节点的子树
        （即 coarse_node_ids[k] 是节点 i 的祖先）。
        这是 models.py 里 coarse prototype 聚合的数学基础。

    - coarse_node_ids : [C_coarse] long tensor
        保存 coarse 节点在全局本体中的原始索引。
        允许 models.py / criterion 在需要时回溯到本体空间。

    - num_coarse_classes : int
        coarse 节点总数，= len(coarse_node_ids)。

    "Coarse 节点"的定义：
        本体 DAG 中深度 depth ∈ [1, coarse_depth_threshold] 的节点。
        depth=0 通常是根节点 ("cell")，太泛化，排除。
        默认 threshold=3，覆盖 lineage 级别的粗粒度分类（如 T-cell, B-cell 等谱系）。
        可通过 __init__ 参数调整。
    """

    def __init__(self, compiled_path, device='cpu', coarse_depth_threshold=3):
        """
        Args:
            compiled_path         : 预编译的 .pt 文件路径
            device                : 运算设备
            coarse_depth_threshold: 深度 <= 此值的节点被认定为 coarse 节点 (默认 3)
        """
        print(f"[Utils] Loading compiled ontology from {compiled_path}...")
        checkpoint = torch.load(compiled_path, map_location=device)

        # ── 基础映射 ──────────────────────────────────────────────
        self.device = device
        self.id2idx   = checkpoint['mappings']['id2idx']
        self.term_map = checkpoint['mappings']['term_map']
        self.all_terms = checkpoint['structure']['all_terms']

        # ── 结构矩阵 ──────────────────────────────────────────────
        # ancestor_matrix[i, j] = True 表示 j 是 i 的祖先（含自身）
        self.ancestor_matrix = checkpoint['structure']['ancestor_matrix_incl_self'].to(device).bool()
        self.depths           = checkpoint['structure']['depths'].to(device)  # [N]

        self.num_classes = len(self.all_terms)  # 全局本体节点总数 N

        # ── 新增：构建 fine→coarse 归属掩码 ───────────────────────
        # 这一步是连接 OntologyManager ↔ models.py 的关键桥梁
        self.fine_to_coarse_mask, self.coarse_node_ids = \
            self._build_fine_to_coarse_mask(coarse_depth_threshold)

        self.num_coarse_classes = len(self.coarse_node_ids)

        print(f"[Utils] Ontology loaded: {self.num_classes} nodes total, "
              f"{self.num_coarse_classes} coarse nodes (depth ≤ {coarse_depth_threshold})")

        self._build_id2name()      

    def _build_id2name(self):
        """惰性建立反向映射，優先使用可讀的自然語言名稱"""
        if hasattr(self, 'id2name') and self.id2name is not None:
            return
        
        print("[OntologyManager] Building id2name with readable name priority...")
        
        id2name = {}
        for name, idx in self.term_map.items():
            if idx not in id2name or \
            (str(id2name[idx]).upper().startswith('CL:') and 
                not str(name).upper().startswith('CL:')):
                id2name[idx] = name
        
        self.id2name = id2name
        print(f"[OntologyManager] id2name built successfully: {len(id2name)} entries")

    # ──────────────────────────────────────────────────────────────
    # 私有方法：构建 fine_to_coarse_mask
    # ──────────────────────────────────────────────────────────────
    def _build_fine_to_coarse_mask(self, coarse_depth_threshold: int):
        """
        从 ancestor_matrix 和 depths 派生 fine→coarse 归属掩码。

        数学含义：
            fine_to_coarse_mask[i, k] = True
            ⟺ coarse_node_ids[k] 是节点 i 的祖先
            ⟺ 节点 i 在 coarse 节点 k 所代表的细胞谱系下

        实现步骤：
            Step 1 — 确定 coarse 节点集合：depths ∈ [1, threshold] 的节点
            Step 2 — 从 ancestor_matrix 中提取对应列：anc[:, coarse_ids]
            Step 3 — 排除"节点是自身 coarse 祖先"的对角情况

        Args:
            coarse_depth_threshold: int，深度上限

        Returns:
            fine_to_coarse_mask : [N, C_coarse] bool Tensor，存于 self.device
            coarse_node_ids     : [C_coarse] long Tensor，存于 self.device
        """
        depths_cpu = self.depths.cpu()          # 在 CPU 上操作，避免 GPU OOM
        N = self.ancestor_matrix.shape[0]

        # ── Step 1: 确定 coarse 节点 ──────────────────────────────
        # 深度 0 = 根节点（如 "cell"），太泛，排除
        # 深度 ∈ [1, threshold] = lineage / 粗粒度类别层，保留
        coarse_bool_mask = (depths_cpu > 0) & (depths_cpu <= coarse_depth_threshold)
        coarse_ids = torch.where(coarse_bool_mask)[0]   # [C_coarse]，CPU long tensor

        if len(coarse_ids) == 0:
            raise RuntimeError(
                f"[Utils] No coarse nodes found with depth ≤ {coarse_depth_threshold}. "
                f"Try increasing coarse_depth_threshold. "
                f"Depth distribution: min={int(depths_cpu.min())}, "
                f"max={int(depths_cpu.max())}, "
                f"median={float(depths_cpu.float().median()):.1f}"
            )

        # ── Step 2: 切片 ancestor_matrix ──────────────────────────
        # ancestor_matrix 可能很大（N×N），全量转 CPU dense 再切片
        # 若 ancestor_matrix 本身是稀疏格式需先 to_dense()
        anc = self.ancestor_matrix.cpu()        # [N, N] bool，CPU

        # 直接按列索引：取 coarse 节点对应列
        # fine_to_coarse[i, k] = anc[i, coarse_ids[k]]
        # 即：节点 i 的祖先集合是否包含第 k 个 coarse 节点
        fine_to_coarse = anc[:, coarse_ids]     # [N, C_coarse] bool，CPU

        # ── Step 3: 排除自身（向量化，不用循环）─────────────────────
        # 若节点 i 本身也是某个 coarse 节点 coarse_ids[k]，
        # 则 ancestor_matrix 含自身（incl_self），会把 fine_to_coarse[i, k] 标为 True
        # 但 coarse prototype 的语义是"父节点"，不应包含自身
        # 构造布尔掩码：self_mask[i, k] = True 当且仅当 i == coarse_ids[k]
        #
        # 等价实现：在 [N, C_coarse] 空间里，只有 (coarse_ids[k], k) 位置为 True
        C_coarse = len(coarse_ids)
        k_local  = torch.arange(C_coarse)       # [0, 1, ..., C_coarse-1]
        self_mask = torch.zeros(N, C_coarse, dtype=torch.bool)
        # coarse_ids[k] 是第 k 个 coarse 节点的全局 index
        # 若节点 coarse_ids[k] 存在于 fine 空间，排除其"视自己为祖先"
        valid = coarse_ids < N                  # 防御性检查（理论上永远 True）
        self_mask[coarse_ids[valid], k_local[valid]] = True

        fine_to_coarse = fine_to_coarse & ~self_mask    # 排除自身后的最终掩码

        # ── 合法性断言 ────────────────────────────────────────────
        # 检查是否每个 coarse 节点至少有一个 fine 子节点，否则该 coarse 节点是孤立的
        has_children = fine_to_coarse.any(dim=0)        # [C_coarse] bool
        n_empty = (~has_children).sum().item()
        if n_empty > 0:
            print(f"[Utils] Warning: {n_empty}/{C_coarse} coarse nodes have no fine children "
                  f"(they may be leaf nodes at depth ≤ {coarse_depth_threshold}). "
                  f"Consider adjusting coarse_depth_threshold.")

        return fine_to_coarse.to(self.device), coarse_ids.to(self.device)

    # ──────────────────────────────────────────────────────────────
    # 公共方法（与旧版完全兼容）
    # ──────────────────────────────────────────────────────────────
    def map_labels(self, labels):
        """
        将字符串标签映射到本体全局索引。

        Returns:
            clean_indices : [n] long Tensor，-1 替换为 0（防止越界）
            mask          : [n] bool Tensor，True = 映射成功
        """
        raw_indices = [self.term_map.get(norm(l), -1) for l in labels]
        indices = torch.tensor(raw_indices, dtype=torch.long, device=self.device)
        mask = (indices != -1).to(self.device)

        clean_indices = indices.clone()
        clean_indices[~mask] = 0
        return clean_indices, mask

    def get_ancestor_mask(self, batch_labels, mask=None):
        if mask is not None:
            batch_labels = batch_labels[mask]
        return self.ancestor_matrix[batch_labels]

    def get_hierarchy_weights(self):
        """
        返回每个类别的深度加权系数，用于 L_HCL 权重项。
        越深（越具体）的细胞，权重越高。
        """
        d = self.depths.clamp(min=0).float()
        w = torch.log1p(d)
        return w / w.max()

    def get_hierarchy_margins(self, base_margin=0.1, max_margin=0.5):
        """
        根据节点深度计算动态 Margin（对应论文创新点：早期大 m，后期小 m）。
        深层节点 Margin 更大（细粒度分辨更难，需要更严格约束）。
        """
        d = torch.log1p(self.depths.clamp(min=0).float())
        d = d / d.max()
        return torch.clamp(base_margin * (1 + d), max=max_margin)

    def get_fine_to_coarse_for_src(self, src_ids: torch.Tensor):
        """
        [新增便捷方法] 为 train.py 中的 unique_src_ids 切片对应的子矩阵。

        在 models.py 里，prototype_head 的行对应所有 N 个本体节点，
        但训练时实际使用的只有 unique_src_ids 这些行。
        此方法返回只针对这些 src fine 节点的 fine→coarse 子矩阵，
        避免 models.py 传入整个 N×C_coarse 矩阵（大多是无效行）。

        Args:
            src_ids: [n_src] long Tensor，unique_src_ids（训练中实际出现的本体 ID）

        Returns:
            sub_mask: [n_src, C_coarse] bool Tensor
                      sub_mask[i, k] = True 表示第 i 个 src 类属于第 k 个 coarse 节点的子树
        """
        src_ids_cpu = src_ids.cpu()
        return self.fine_to_coarse_mask[src_ids_cpu].to(self.device)


# ──────────────────────────────────────────────────────────────────
# DataDict（与旧版完全一致，无修改）
# ──────────────────────────────────────────────────────────────────
class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        if self.size == 0:
            return self
        shuffled = DataDict()
        idx = random_state.permutation(self.size)
        for k, v in self.items():
            shuffled[k] = v[idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray, torch.Tensor)):
            return DataDict([(k, v[fetch]) for k, v in self.items()])
        return super().__getitem__(fetch)

    def get_batch(self, batch_size, start_idx=0, device='cpu'):
        end_idx = min(start_idx + batch_size, self.size)
        fetch = slice(start_idx, end_idx)
        batch_data = self[fetch]
        if 'X' in batch_data:
            batch_data['X'] = torch.tensor(batch_data['X'], dtype=torch.float, device=device)
        return batch_data


# ──────────────────────────────────────────────────────────────────
# 稀疏/编码工具（与旧版完全一致，无修改）
# ──────────────────────────────────────────────────────────────────
def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper  = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower  = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr  = empty_safe(np.vectorize(str), str)