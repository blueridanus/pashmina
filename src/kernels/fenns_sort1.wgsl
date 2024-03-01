struct Params {
    cell_width: f32,
    search_radius: f32,
}

@group(0) @binding(0)
var<uniform> params: Params;

struct Particle {
    position: vec3f,
}

@group(0) @binding(1)
var<storage, read> input: array<Particle>;

@group(0) @binding(2)
var<storage, read_write> count: array<atomic<u32>>;

const GRID_DIM: u32 = 18;
const GRID_SIZE: u32 = GRID_DIM * GRID_DIM * GRID_DIM;
var<workgroup> shCount: array<atomic<u32>, GRID_SIZE>;

const WG_SIZE: u32 = 64;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    if global_id.x < arrayLength(&input) {
        let particle = input[global_id.x];
        let grid_pos = vec3u(particle.position / params.cell_width);
        let grid_cell_idx = grid_pos.z * GRID_DIM * GRID_DIM + grid_pos.y * GRID_DIM + grid_pos.x;
        atomicAdd(&shCount[grid_cell_idx], 1u);
    }
    workgroupBarrier();

    for (var i = 0u; i <= GRID_SIZE / WG_SIZE; i += 1u) {
        let offset = i * WG_SIZE + local_id.x;
        if offset < GRID_SIZE {
            let particleCount = atomicLoad(&shCount[offset]);
            atomicAdd(&count[offset], particleCount);
        }
    }
}