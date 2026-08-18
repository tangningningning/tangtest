# -*- coding: utf-8 -*-
"""
无线信道模型。

当前实现：
1. TV -> SV：当前地面大尺度信道；
2. TV -> RSU：当前地面链路实现；
3. TV -> UAV：概率 LoS / NLoS 平均空地信道。
所有参数统一从 config.py 读取。
"""

import math
import config


def _dbm_to_w(dbm: float) -> float:
    """dBm 转 W。"""
    return 10 ** ((dbm - 30.0) / 10.0)


def _noise_power_w() -> float:
    """计算接收机噪声功率。"""
    noise_dbm = (
        config.NOISE_PSD_DBM_HZ
        + 10.0 * math.log10(config.BANDWIDTH_HZ)
        + config.NOISE_FIGURE_DB
    )
    return _dbm_to_w(noise_dbm)


NOISE_POWER_W = _noise_power_w()


# =============================================================================
# TV -> SV
# =============================================================================

def sv_link_eligible(distance_m: float) -> bool:
    """判断 V2V 链路是否允许进入 P3 候选集。"""
    return distance_m <= config.V2V_MAX_DISTANCE_M


def sv_full_rate_bps(distance_m: float) -> float:
    """TV -> SV 在 b=1 时的基础速率 R_bar。"""
    if not sv_link_eligible(distance_m):
        raise ValueError(
            f"SV 链路距离 {distance_m:.3f} m 超过 "
            f"{config.V2V_MAX_DISTANCE_M:.3f} m"
        )

    d_eff = max(distance_m, config.V2V_DISTANCE_FLOOR_M)
    pl_db = (
        config.V2V_PL0_DB
        + 10.0 * config.V2V_PATHLOSS_EXPONENT * math.log10(d_eff)
    )
    gain = 10 ** (-pl_db / 10.0)
    snr = config.TX_POWER_W * gain / NOISE_POWER_W
    return config.BANDWIDTH_HZ * math.log2(1.0 + snr)



# =============================================================================
# TV -> RSU
# =============================================================================

def rsu_link_eligible(distance_m: float) -> bool:
    """
    当前 RSU 主线集成使用与 SV 相同的地面链路有效范围。

    注意：
    这是当前统一地面链路 pilot 的实现选择，
    最终实验参数校准时应再单独检查 RSU/V2I 参数。
    """
    return distance_m <= config.V2V_MAX_DISTANCE_M


def rsu_full_rate_bps(distance_m: float) -> float:
    """
    TV -> RSU 在 b=1 时的基础速率 R_bar。

    当前复用已有地面路径损耗表达式：
        PL(d) = PL0 + 10*alpha*log10(d)

    这里只复用数学/数值实现，
    不把当前 V2V 实测有效性结论扩展成新的 RSU 实测结论。
    """
    if not rsu_link_eligible(distance_m):
        raise ValueError(
            f"RSU 链路距离 {distance_m:.3f} m 超过当前 pilot 地面范围 "
            f"{config.V2V_MAX_DISTANCE_M:.3f} m"
        )

    d_eff = max(distance_m, config.V2V_DISTANCE_FLOOR_M)
    pl_db = (
        config.V2V_PL0_DB
        + 10.0 * config.V2V_PATHLOSS_EXPONENT * math.log10(d_eff)
    )
    gain = 10 ** (-pl_db / 10.0)
    snr = config.TX_POWER_W * gain / NOISE_POWER_W
    return config.BANDWIDTH_HZ * math.log2(1.0 + snr)


# =============================================================================
# TV -> UAV
# =============================================================================

def uav_los_probability(horizontal_distance_m: float) -> float:
    """
    计算 TV -> UAV 的 LoS 概率。

    固定高度 H，水平距离 r：
        theta = atan(H/r)
        P_LoS = 1/[1+a*exp(-b(theta-a))]

    theta 在公式中使用度。
    """
    r = max(float(horizontal_distance_m), 0.0)

    if r <= 1e-12:
        theta_deg = 90.0
    else:
        theta_deg = math.degrees(math.atan(config.UAV_HEIGHT_M / r))

    return 1.0 / (
        1.0
        + config.UAV_LOS_A
        * math.exp(-config.UAV_LOS_B * (theta_deg - config.UAV_LOS_A))
    )


def uav_mean_pathloss_db(horizontal_distance_m: float) -> float:
    """
    概率 LoS/NLoS 平均空地路径损耗。

    d = sqrt(r^2 + H^2)
    FSPL = 20 log10(4*pi*f_c*d/c)
    PL = FSPL + P_LoS*eta_LoS + (1-P_LoS)*eta_NLoS
    """
    r = max(float(horizontal_distance_m), 0.0)
    d_3d = math.sqrt(r * r + config.UAV_HEIGHT_M ** 2)

    fspl_db = 20.0 * math.log10(
        4.0
        * math.pi
        * config.UAV_CARRIER_FREQ_HZ
        * d_3d
        / config.SPEED_OF_LIGHT_MPS
    )

    p_los = uav_los_probability(r)
    extra_loss_db = (
        p_los * config.UAV_ETA_LOS_DB
        + (1.0 - p_los) * config.UAV_ETA_NLOS_DB
    )

    return fspl_db + extra_loss_db


def uav_full_rate_bps(horizontal_distance_m: float) -> float:
    """TV -> UAV 在 b=1 时的基础速率 R_bar。"""
    pl_db = uav_mean_pathloss_db(horizontal_distance_m)
    gain = 10 ** (-pl_db / 10.0)
    snr = config.TX_POWER_W * gain / NOISE_POWER_W
    return config.BANDWIDTH_HZ * math.log2(1.0 + snr)


# =============================================================================
# P3 统一信道接口
# =============================================================================

def full_rate_bps(node_type: str, distance_m: float) -> float:
    """
    P3 统一基础速率接口。

    distance_m 的含义：
    - SV：二维地面距离；
    - UAV：TV 到 UAV 投影点的水平距离。
    """
    node_type = node_type.upper()

    if node_type == "SV":
        return sv_full_rate_bps(distance_m)

    if node_type == "UAV":
        return uav_full_rate_bps(distance_m)

    if node_type == "RSU":
        return rsu_full_rate_bps(
            distance_m
        )

    raise ValueError(f"未知节点类型: {node_type}")
