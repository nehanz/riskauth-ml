"""Network behaviour feature computation for RiskAuth v2."""


def asn_change_flag(asn: str, known_asns: list[str]) -> float:
    """1.0 if ASN has never been seen for this user."""
    if not known_asns:
        return 0.0
    return 0.0 if asn in known_asns else 1.0


def ip_reputation_score() -> float:
    """
    Placeholder for external IP reputation service.
    In production, query a threat-intelligence API.
    Returns 0.5 (neutral) by default.
    """
    return 0.5


def rtt_deviation(current_rtt: float, avg_rtt: float) -> float:
    """Normalised deviation of RTT from user baseline."""
    if avg_rtt <= 0:
        return 0.0
    return abs(current_rtt - avg_rtt) / (avg_rtt + 1e-6)


def network_frequency_score(asn: str, asn_freq: dict[str, int]) -> float:
    """How common this ASN is for the user. 1.0 = most common."""
    if not asn_freq:
        return 1.0
    total = sum(asn_freq.values())
    count = asn_freq.get(asn, 0)
    return count / total if total > 0 else 0.0
