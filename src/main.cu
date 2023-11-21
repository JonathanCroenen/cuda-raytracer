#include "camera.h"
#include "scene.h"
#include "renderer.h"
#include "scenes/cornell_quads.h"
#include "utils/ppm.h"

using namespace rtx;

int main() {
    using vec3 = math::vec3<float>;

    Renderer::Parameters params = {
        .width = 800,
        .height = 600,
        .rays_per_pixel = 1000,
        .max_depth = 10,
        .num_threads = dim3(16 , 16),
    };

    auto renderer = Renderer::create(params);
    auto scene = scene::create_cornell_quads();

    auto camera = Camera::create(vec3(0.0f, 0.0f, 3.0f), vec3(0.0f, 0.0f, -1.0f),
                                 vec3(0.0f, 1.0f, 0.0f), 90.0f,
                                 float(params.width) / float(params.height));

    vec3* framebuffer = renderer->render(camera.get(), scene.get());

    print_ppm(framebuffer, params.width, params.height);

    return 0;
}
