#include "math/vec3.h"


namespace rtx {

struct LightSample {
private:
    using vec3 = math::vec3<float>;

public:
    vec3 position;
    vec3 color;
    float intensity;
};


}
