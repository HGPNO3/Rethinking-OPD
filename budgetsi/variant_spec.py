"""The five variants previously prioritized with HG; no experiment auto-launch."""

from dataclasses import asdict, dataclass

SCHEMA = "budgetsi_upstream_variants_online_v1"


@dataclass(frozen=True)
class Variant:
    name: str
    priority: int
    k: int
    strategy: str
    weight: str

    def contract(self):
        return asdict(self)


VARIANTS = {
    v.name: v
    for v in (
        Variant("student_top16", 1, 16, "only_stu", "student_p"),
        Variant("union_student", 2, 16, "union", "student_p"),
        Variant("intersection_student", 3, 16, "intersection", "student_p"),
        Variant("sampled_token", 4, 0, "only_stu", "student_p"),
        Variant("union_teacher", 5, 16, "union", "teacher_p"),
    )
}
DEFAULT = VARIANTS["student_top16"]


def get_variant(name):
    try:
        return VARIANTS[name]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Unknown OPD variant: {name!r}") from e


def from_config(config):
    if config.get("schema_version") == SCHEMA:
        variant = get_variant(config.get("opd_variant"))
        if config.get("opd") != variant.contract():
            raise ValueError("Variant name and algorithm contract differ")
        return variant
    if "opd_variant" in config or "opd" in config:
        raise ValueError("Explicit variants require the variants schema")
    if config.get("schema_version") != "budgetsi_upstream_top16_online_v1":
        raise ValueError("Unsupported OPD configuration schema")
    return DEFAULT
