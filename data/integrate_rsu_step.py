# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

CONFIG = SRC / "config.py"
CHANNEL = SRC / "channel.py"
CACHE = SRC / "cache_model.py"
P3 = SRC / "p3_solver.py"
RSU_STATE = SRC / "rsu_state.py"
DEPLOYMENT = ROOT / "data" / "rsu" / "rsu_nodes.csv"

for path in (CONFIG, CHANNEL, CACHE, P3, RSU_STATE, DEPLOYMENT):
    if not path.is_file():
        raise FileNotFoundError(f"缺少必要文件：{{path}}")

config_text = CONFIG.read_text(encoding="utf-8")
channel_text = CHANNEL.read_text(encoding="utf-8")
cache_text = CACHE.read_text(encoding="utf-8")
p3_text = P3.read_text(encoding="utf-8")

if "RSU_CPU_HZ" not in config_text:
    raise RuntimeError(
        "config.py 中不存在 RSU_CPU_HZ；"
        "本步骤不会凭空新增 CPU 参数。"
    )

if "RSU_DEPLOYMENT_FILE" not in config_text:
    config_text += "\n\n# =============================================================================\n"
    config_text += "# RSU 固定部署与缓存\n"
    config_text += "# =============================================================================\n\n"
    config_text += "RSU_DEPLOYMENT_FILE = DATA_DIR / \"rsu\" / \"rsu_nodes.csv\"\n"
    config_text += "RSU_CACHE_CAPACITY_RATIO = SV_CACHE_CAPACITY_RATIO\n"

if "def rsu_full_rate_bps(" not in channel_text:
    marker = (
        "# =============================================================================\n"
        "# TV -> UAV\n"
        "# ============================================================================="
    )
    if marker not in channel_text:
        raise RuntimeError("channel.py 中未找到 TV -> UAV 代码段。")
    channel_text = channel_text.replace(
        marker,
        '\n# =============================================================================\n# TV -> RSU\n# =============================================================================\n\ndef rsu_link_eligible(distance_m: float) -> bool:\n    """\n    RSU 与 SV 同属于当前地面链路模型。\n\n    当前不新增未经验证的 RSU 覆盖半径，\n    沿用现有地面模型的有效距离范围。\n    """\n    return sv_link_eligible(distance_m)\n\n\ndef rsu_full_rate_bps(distance_m: float) -> float:\n    """\n    TV -> RSU 满资源速率。\n\n    当前系统模型中 TV->SV 与 TV->RSU\n    使用同一地面链路形式，因此直接复用\n    现有地面速率实现。\n    """\n    if not rsu_link_eligible(distance_m):\n        raise ValueError(\n            f"RSU 地面链路距离 {distance_m:.3f} m "\n            "超出当前地面模型有效范围。"\n        )\n\n    return sv_full_rate_bps(distance_m)\n\n\n' + marker,
        1,
    )

full_rate_start = channel_text.find("def full_rate_bps(")
if full_rate_start < 0:
    raise RuntimeError("channel.py 中缺少 full_rate_bps。")

if "return rsu_full_rate_bps(" not in channel_text[full_rate_start:]:
    pattern = re.compile(
        r'    if node_type == "RSU":\n'
        r'        raise NotImplementedError\(\n'
        r'.*?'
        r'        \)\n',
        flags=re.S,
    )
    replacement = (
        '    if node_type == "RSU":\n'
        '        return rsu_full_rate_bps(\n'
        '            distance_m\n'
        '        )\n'
    )
    channel_text, count = pattern.subn(
        replacement,
        channel_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("未能唯一替换 channel.py 的 RSU 分支。")

if "def rsu_cache_capacity_mb(" not in cache_text:
    marker = "\ndef node_cache_capacity_mb("
    if marker not in cache_text:
        raise RuntimeError("cache_model.py 中缺少 node_cache_capacity_mb。")
    cache_text = cache_text.replace(
        marker,
        '\ndef rsu_cache_capacity_mb(\n    service_sizes_mb: dict[int, float],\n) -> float:\n    """RSU 集成阶段缓存容量。"""\n    return (\n        config.RSU_CACHE_CAPACITY_RATIO\n        * sum(service_sizes_mb.values())\n    )\n\n\n' + marker,
        1,
    )

old_branch = (
    '    if node_type == "RSU":\n'
    '        raise NotImplementedError(\n'
    '            "RSU 缓存容量尚未正式配置。"\n'
    '        )\n'
)
if old_branch in cache_text:
    cache_text = cache_text.replace(
        old_branch,
        '    if node_type == "RSU":\n'
        '        return rsu_cache_capacity_mb(\n'
        '            service_sizes_mb\n'
        '        )\n',
        1,
    )
elif (
    'if node_type == "RSU":' not in cache_text
    or 'return rsu_cache_capacity_mb(' not in cache_text
):
    raise RuntimeError("未能确认 cache_model.py 的 RSU 容量分支。")

if "import rsu_state" not in p3_text:
    if "import cache_model" in p3_text:
        p3_text = p3_text.replace(
            "import cache_model",
            "import cache_model\nimport rsu_state",
            1,
        )
    else:
        raise RuntimeError("p3_solver.py 中缺少 import cache_model。")

pattern = re.compile(
    r"def build_edge_nodes\(.*?(?=\n\ndef build_all_service_cache_state\()",
    flags=re.S,
)
p3_text, count = pattern.subn(
    'def build_edge_nodes(\n    uav_positions:\n        dict[str, tuple[float, float]]\n        | None = None,\n) -> list[dict]:\n    """\n    构造完整边缘节点集合：\n        M = V_S ∪ R ∪ U\n    """\n    nodes = build_current_sv_nodes()\n\n    nodes.extend(\n        rsu_state.load_rsu_nodes()\n    )\n\n    if uav_positions:\n        nodes.extend(\n            build_uav_nodes(uav_positions)\n        )\n\n    return nodes\n',
    p3_text,
    count=1,
)
if count != 1:
    raise RuntimeError("未能唯一更新 build_edge_nodes。")

pattern = re.compile(
    r"def _node_position\(.*?(?=\n\ndef make_edge_candidate\()",
    flags=re.S,
)
p3_text, count = pattern.subn(
    'def _node_position(\n    slot: int,\n    node: dict,\n    positions: dict,\n) -> tuple[float, float]:\n    """获取边缘节点二维水平位置。"""\n    node_type = node["node_type"].upper()\n\n    if node_type == "SV":\n        key = (slot, node["node_id"])\n\n        if key not in positions:\n            raise RuntimeError(f"缺少 SV 位置: {key}")\n\n        return positions[key]\n\n    if node_type in {"RSU", "UAV"}:\n        if "position" not in node:\n            raise RuntimeError(\n                f"{node_type} {node[\'node_id\']} 缺少 position。"\n            )\n\n        return (\n            float(node["position"][0]),\n            float(node["position"][1]),\n        )\n\n    raise NotImplementedError(\n        f"{node_type} 位置接口尚未接入。"\n    )\n',
    p3_text,
    count=1,
)
if count != 1:
    raise RuntimeError("未能唯一更新 _node_position。")

if '    # TV 到节点投影点的二维水平距离。\n    horizontal_distance_m = math.hypot(tx - nx, ty - ny)\n\n    if node_type == "SV":\n        if not channel.sv_link_eligible(horizontal_distance_m):\n            return None\n        link_distance_m = horizontal_distance_m\n\n    elif node_type == "UAV":\n        # UAV 的真实无线链路长度是三维距离。\n        # 信道函数输入水平距离，并在内部加入固定高度 H。\n        link_distance_m = math.sqrt(\n            horizontal_distance_m ** 2 + config.UAV_HEIGHT_M ** 2\n        )\n\n    else:\n        link_distance_m = horizontal_distance_m\n\n    # ---------- 3. 满资源基础速率 ----------\n    rbar_bps = channel.full_rate_bps(\n        node_type,\n        horizontal_distance_m,\n    )\n' in p3_text:
    p3_text = p3_text.replace(
        '    # TV 到节点投影点的二维水平距离。\n    horizontal_distance_m = math.hypot(tx - nx, ty - ny)\n\n    if node_type == "SV":\n        if not channel.sv_link_eligible(horizontal_distance_m):\n            return None\n        link_distance_m = horizontal_distance_m\n\n    elif node_type == "UAV":\n        # UAV 的真实无线链路长度是三维距离。\n        # 信道函数输入水平距离，并在内部加入固定高度 H。\n        link_distance_m = math.sqrt(\n            horizontal_distance_m ** 2 + config.UAV_HEIGHT_M ** 2\n        )\n\n    else:\n        link_distance_m = horizontal_distance_m\n\n    # ---------- 3. 满资源基础速率 ----------\n    rbar_bps = channel.full_rate_bps(\n        node_type,\n        horizontal_distance_m,\n    )\n',
        '    # TV 到节点投影点的二维水平距离。\n    horizontal_distance_m = math.hypot(\n        tx - nx,\n        ty - ny,\n    )\n\n    if node_type == "SV":\n        if not channel.sv_link_eligible(horizontal_distance_m):\n            return None\n        link_distance_m = horizontal_distance_m\n\n    elif node_type == "RSU":\n        if not channel.rsu_link_eligible(horizontal_distance_m):\n            return None\n        link_distance_m = horizontal_distance_m\n\n    elif node_type == "UAV":\n        link_distance_m = math.sqrt(\n            horizontal_distance_m ** 2\n            + config.UAV_HEIGHT_M ** 2\n        )\n\n    else:\n        link_distance_m = horizontal_distance_m\n\n    # ---------- 3. 满资源基础速率 ----------\n    rbar_bps = channel.full_rate_bps(\n        node_type,\n        horizontal_distance_m,\n    )\n',
        1,
    )
else:
    start = p3_text.find("def make_edge_candidate(")
    if start < 0 or 'elif node_type == "RSU":' not in p3_text[start:]:
        raise RuntimeError("未找到当前 P3 的距离/速率代码块。")

for text, path in (
    (config_text, CONFIG),
    (channel_text, CHANNEL),
    (cache_text, CACHE),
    (p3_text, P3),
):
    compile(text, str(path), "exec")

CONFIG.write_text(config_text, encoding="utf-8")
CHANNEL.write_text(channel_text, encoding="utf-8")
CACHE.write_text(cache_text, encoding="utf-8")
P3.write_text(p3_text, encoding="utf-8")

print("=" * 72)
print("RSU 接入完成")
print("=" * 72)
print("边缘节点集合      : SV + RSU + UAV")
print("RSU 位置          : data/rsu/rsu_nodes.csv")
print("RSU 信道          : 复用当前地面链路模型")
print("RSU CPU           : 沿用 config.RSU_CPU_HZ")
print("RSU 缓存容量比例  : 等于 SV_CACHE_CAPACITY_RATIO")
print()
print("未修改：P1 / P2 / Pjoint / async_update / rolling_controller / Cloud")

try:
    Path(__file__).unlink()
    print("临时补丁脚本已删除。")
except OSError:
    print("临时补丁未能自动删除，请手动删除该文件。")
