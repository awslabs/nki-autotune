"""Default schedule fixtures for schedule tests."""

from nkigym.schedule.types import DimSchedule, Schedule

MATMUL_256_DEFAULT = Schedule(
    loop_order=(("d1", 0), ("d3", 0), ("d0", 0)),
    dim_schedules=(DimSchedule("d1", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 1)),
    op_placements=(2, 2),
)


MATMUL_RECT_DEFAULT = Schedule(
    loop_order=(("d1", 0), ("d3", 0), ("d0", 0)),
    dim_schedules=(DimSchedule("d1", 128, 1), DimSchedule("d3", 512, 1), DimSchedule("d0", 128, 1)),
    op_placements=(2, 2),
)


MATMUL_TANH_DEFAULT = Schedule(
    loop_order=(("d1", 0), ("d3", 0), ("d0", 0)),
    dim_schedules=(DimSchedule("d1", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 1)),
    op_placements=(2, 2),
)


ADD_ONLY_DEFAULT = Schedule(
    loop_order=(("d0", 0), ("d1", 0)),
    dim_schedules=(DimSchedule("d0", 128, 1), DimSchedule("d1", 128, 1)),
    op_placements=(2, 2),
)


MATMUL_ADD_DEFAULT = Schedule(
    loop_order=(("d1", 0), ("d3", 0), ("d0", 0)),
    dim_schedules=(DimSchedule("d1", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d0", 128, 1)),
    op_placements=(2, 2, 2),
)


RMSNORM_MATMUL_DEFAULT = Schedule(
    loop_order=(("d0", 0), ("d3", 0), ("d1", 0), ("d1", 1)),
    dim_schedules=(DimSchedule("d0", 128, 1), DimSchedule("d3", 256, 1), DimSchedule("d1", 128, 1)),
    op_placements=(2, 2),
)
