struct Params {
    cell_width: f32,
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
var<workgroup> shared_count: array<atomic<u32>, GRID_SIZE>;

const WG_SIZE: u32 = 64;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    if global_id.x < arrayLength(&input) {
        let particle = input[global_id.x];
        let pos_v = vec3u(particle.position);
        let pos = pos_v.z * GRID_DIM * GRID_DIM + pos_v.y * GRID_DIM + pos_v.x;
        atomicAdd(&shared_count[pos], 1u);
    }
    workgroupBarrier();

    for (var i = 0u; i <= GRID_SIZE / WG_SIZE; i += 1u) {
        let offset = i * WG_SIZE + local_id.x;
        if offset < GRID_SIZE {
            let particle_count = atomicLoad(&shared_count[offset]);
            atomicAdd(&count[offset], particle_count);
        }
    }
}