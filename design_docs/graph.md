# ComputeGraph Dimension Tracking Architecture

## Overview

## Tensor Tracking
The computation is broken into parallel tiles based on operator semantics and tile sizes along the parallel axes. Parallel counter only affects the load and store happening at subgraph boundaries with HBM. The subgraph internal SBUF/PSUM tensors do not have axes coordinates to track but they only have sizes.