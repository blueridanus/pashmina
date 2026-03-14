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

const WG_SIZE: u32 = 64;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    if global_id.x >= arrayLength(&input) {
        return;
    }

    let particle = input[global_id.x];
    let grid_pos = vec3u(particle.position / params.cell_width);
    let grid_cell_idx = grid_pos.z * GRID_DIM * GRID_DIM + grid_pos.y * GRID_DIM + grid_pos.x;
    atomicAdd(&count[grid_cell_idx], 1u);
}
