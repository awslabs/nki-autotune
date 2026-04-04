"""Golden data for combinatorial schedule enumeration tests."""

from nkigym.schedule.types import DimSchedule, Schedule

MATMUL_256_LOOP_ORDERS = [
    (("d1", 0), ("d3", 0), ("d0", 0)),
    (("d1", 0), ("d0", 0), ("d3", 0)),
    (("d3", 0), ("d1", 0), ("d0", 0)),
    (("d3", 0), ("d0", 0), ("d1", 0)),
    (("d0", 0), ("d1", 0), ("d3", 0)),
    (("d0", 0), ("d3", 0), ("d1", 0)),
]

MATMUL_256_OP_PLACEMENTS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

MATMUL_256_BLOCKINGS = [
    (DimSchedule("d1", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 1)),
    (DimSchedule("d1", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 2)),
    (DimSchedule("d1", 128, 2), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 1)),
    (DimSchedule("d1", 128, 2), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 2)),
]

ADD_ONLY_LOOP_ORDERS = [(("d0", 0), ("d1", 0)), (("d1", 0), ("d0", 0))]

ADD_ONLY_BLOCKINGS = [
    (DimSchedule("d0", 128, 1), DimSchedule("d1", 128, 1)),
    (DimSchedule("d0", 128, 1), DimSchedule("d1", 128, 2)),
    (DimSchedule("d0", 128, 2), DimSchedule("d1", 128, 1)),
    (DimSchedule("d0", 128, 2), DimSchedule("d1", 128, 2)),
]

MATMUL_256_DEFAULT = Schedule(
    loop_order=(("d1", 0), ("d3", 0), ("d0", 0)),
    dim_schedules=(DimSchedule("d1", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 1)),
    op_placements=(2, 2),
)

ADD_ONLY_DEFAULT = Schedule(
    loop_order=(("d0", 0), ("d1", 0)),
    dim_schedules=(DimSchedule("d0", 128, 1), DimSchedule("d1", 128, 1)),
    op_placements=(2, 2),
)

ATTENTION_DEFAULT = Schedule(
    loop_order=(("d0", 0), ("d5", 0), ("d1", 0), ("d2", 0), ("d2", 1), ("d2", 2)),
    dim_schedules=(
        DimSchedule("d0", 128, 1),
        DimSchedule("d5", 128, 1),
        DimSchedule("d1", 128, 1),
        DimSchedule("d2", 512, 1),
    ),
    op_placements=(2, 2, 2),
)
