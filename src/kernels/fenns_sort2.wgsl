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

@group(0) @binding(3)
var<storage, read_write> reordered: array<Particle>;

@group(0) @binding(4)
var<storage, read_write> border_count: array<atomic<u32>>;

const GRID_DIM: u32 = 18;
const GRID_SIZE: u32 = GRID_DIM * GRID_DIM * GRID_DIM;

const WG_SIZE: u32 = 64;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) _local_id: vec3u,
) {
    if global_id.x < arrayLength(&input) {
        let particle = input[global_id.x];
        let gridPos = vec3u(particle.position / params.cell_width);
        let gridCellIdx = gridPos.z * GRID_DIM * GRID_DIM + gridPos.y * GRID_DIM + gridPos.x;

        let innerSize = params.cell_width - 2.0 * params.search_radius;
        let cellCenter = vec3f(params.cell_width / 2.0) + vec3f(gridPos) * params.cell_width;
        let isBorder = any(abs(cellCenter - particle.position) > vec3f(innerSize / 2.0));

        var reorderedPos: u32;
        if isBorder {
            let border_offset = atomicAdd(&border_count[gridCellIdx], 1u);
            reorderedPos = atomicLoad(&count[GRID_SIZE + gridCellIdx]) + border_offset;
        } else {
            reorderedPos = atomicSub(&count[gridCellIdx], 1u) - 1u;
        }

        reordered[reorderedPos] = particle;
    }
}
