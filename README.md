<div align="center">
  <img src="logo.png" alt="Logo" width="500"/>

<h1>
    <strong>
        tittyos
    </strong>
</h1>
</div>

<div align="center">
<strong>A deep learning library built for C++</strong>

</div>

<br></br>

This project is still under development, as such many features are yet to be implemented.

#### Basic Syntax

```rs
#include <tittyos/ty/tittyos.h>
#include <vector>

int main() {
    std::vector<float> data1 = {1, 1, 2, 3, 5};
    std::vector<float> data2 = {2, 3, 5, 7, 11};

    ty::Tensor tensor1(data1, ty::Shape({1, 5}), ty::DType::float32, true); // [1.0, 1.0, 2.0, 3.0, 5.0]
    ty::Tensor tensor2(data2, ty::Shape({1, 5}), ty::DType::float32, true); // [2.0, 3.0, 5.0, 7.0, 11.0]

    ty::Tensor result = ty::sum(ty::multiply(tensor1, tensor2));
    result.backward();

    tensor1.grad(); // [2.0, 3.0, 5.0, 7.0, 11.0]
    tensor2.grad(); // [1.0, 1.0, 2.0, 3.0, 5.0]
}
```

### **Licensing**

- Read [LICENSE.md](https://github.com/SJ1727/tittyos/blob/main/LICENSE)
- Copyright © 2025 Samuel Johnson ([SJ1727](https://github.com/SJ1727))
