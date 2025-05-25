#include <array>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <type_traits>

// Допоміжна структура для двочкового просування типів
template<typename A, typename B>
using Promote = decltype(std::declval<A>() + std::declval<B>());

template<typename... Ts>
struct PromoteMultiple;

template<typename T>
struct PromoteMultiple<T> {
    using type = T;
};

template<typename A, typename B, typename... Rest>
struct PromoteMultiple<A, B, Rest...> {
    using type = typename PromoteMultiple<
        std::common_type_t<A, B>, Rest...
    >::type;
};

template<typename T, std::size_t N>
class Vector {
public:
    using value_type = T;
    static constexpr std::size_t dimension = N;

    Vector() { data_.fill(T{}); }
    explicit Vector(const T& value) { data_.fill(value); }
    Vector(const Vector& other) = default;

    template<typename U>
    Vector(const Vector<U, N>& other) {
        for (std::size_t i = 0; i < N; ++i)
            data_[i] = static_cast<T>(other[i]);
    }

    Vector& operator=(const Vector& other) = default;

    T& operator[](int index) { return data_[normalize_index(index)]; }
    const T& operator[](int index) const { return data_[normalize_index(index)]; }

    constexpr std::size_t size() const noexcept { return N; }

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (std::size_t i = 0; i < N; ++i) {
            os << v.data_[i] << (i + 1 < N ? ", " : "");
        }
        os << "]";
        return os;
    }

    template<typename U>
    auto operator+(const U& scalar) const { return apply_scalar(scalar, std::plus<>{}); }
    template<typename U>
    auto operator-(const U& scalar) const { return apply_scalar(scalar, std::minus<>{}); }
    template<typename U>
    auto operator*(const U& scalar) const { return apply_scalar(scalar, std::multiplies<>{}); }
    template<typename U>
    auto operator/(const U& scalar) const { return apply_scalar(scalar, std::divides<>{}); }

    template<typename U>
    auto operator+(const Vector<U, N>& other) const { return apply_vector(other, std::plus<>{}); }
    template<typename U>
    auto operator-(const Vector<U, N>& other) const { return apply_vector(other, std::minus<>{}); }
    template<typename U>
    auto operator*(const Vector<U, N>& other) const { return apply_vector(other, std::multiplies<>{}); }
    template<typename U>
    auto operator/(const Vector<U, N>& other) const { return apply_vector(other, std::divides<>{}); }

    template<std::size_t M>
    auto resize() const {
        Vector<T, M> result;
        constexpr std::size_t minN = (N < M ? N : M);
        for (std::size_t i = 0; i < minN; ++i)
            result[i] = data_[i];
        return result;
    }

    template<typename U>
    auto convert() const {
        Vector<U, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = static_cast<U>(data_[i]);
        return result;
    }

    template<int StartIdx, int EndIdx>
    auto slice() const {
        constexpr int s = (StartIdx < 0 ? static_cast<int>(N) + StartIdx : StartIdx);
        constexpr int e = (EndIdx   < 0 ? static_cast<int>(N) + EndIdx   : EndIdx);
        static_assert(s >= 0 && s < static_cast<int>(N), "slice start out of range");
        static_assert(e >= 0 && e < static_cast<int>(N), "slice end out of range");
        constexpr std::size_t len = (s <= e ? (e - s + 1) : (s - e + 1));
        Vector<T, len> result;
        if constexpr (s <= e) {
            for (std::size_t i = 0; i < len; ++i)
                result[i] = data_[s + i];
        } else {
            for (std::size_t i = 0; i < len; ++i)
                result[i] = data_[s - i];
        }
        return result;
    }

private:
    std::array<T, N> data_;

    std::size_t normalize_index(int index) const {
        int idx = (index < 0 ? static_cast<int>(N) + index : index);
        if (idx < 0 || idx >= static_cast<int>(N)) {
            std::ostringstream oss;
            oss << "Index " << index << " out of range for Vector<" << N << ">";
            throw std::out_of_range(oss.str());
        }
        return static_cast<std::size_t>(idx);
    }

    template<typename U, typename Op>
    auto apply_scalar(const U& scalar, Op op) const {
        using R = Promote<T, U>;
        Vector<R, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = op(data_[i], scalar);
        return result;
    }

    template<typename U, typename Op>
    auto apply_vector(const Vector<U, N>& other, Op op) const {
        using R = Promote<T, U>;
        Vector<R, N> result;
        for (std::size_t i = 0; i < N; ++i)
            result[i] = op(data_[i], other[i]);
        return result;
    }
};

template<typename T1, std::size_t N, typename U1, typename T2, typename U2>
auto weighted_sum(const Vector<T1, N>& v1,
                  const U1& alpha,
                  const Vector<T2, N>& v2,
                  const U2& beta) {
    using R1 = std::common_type_t<T1, U1>;
    using R2 = std::common_type_t<T2, U2>;
    using R  = typename PromoteMultiple<R1, R2>::type;
    Vector<R, N> result;
    for (std::size_t i = 0; i < N; ++i)
        result[i] = alpha * v1[i] + beta * v2[i];
    return result;
}

template<typename T1, std::size_t N1, typename T2, std::size_t N2>
auto concat(const Vector<T1, N1>& v1, const Vector<T2, N2>& v2) {
    using R = typename PromoteMultiple<T1, T2>::type;
    constexpr std::size_t M = N1 + N2;
    Vector<R, M> result;
    for (std::size_t i = 0; i < N1; ++i) result[i] = v1[i];
    for (std::size_t j = 0; j < N2; ++j) result[N1 + j] = v2[j];
    return result;
}

template<typename V, typename... Vs>
auto concat(const V& first, const Vs&... rest) {
    constexpr std::size_t total = (V::dimension + ... + Vs::dimension);
    using R = typename PromoteMultiple<typename V::value_type, typename Vs::value_type...>::type;
    Vector<R, total> result;
    std::size_t pos = 0;
    auto append = [&](const auto& vec) {
        for (std::size_t i = 0; i < std::decay_t<decltype(vec)>::dimension; ++i)
            result[pos++] = vec[i];
    };
    (append(first), ..., append(rest));
    return result;
}

template<typename T, typename... Args>
auto make_vector(Args&&... args) {
    constexpr std::size_t N = sizeof...(Args);
    Vector<T, N> result;
    std::size_t i = 0;
    ((result[i++] = static_cast<T>(args)), ...);
    return result;
}

template<typename... Args>
auto build_vector(Args&&... args) {
    using U = std::common_type_t<Args...>;
    constexpr std::size_t N = sizeof...(Args);
    Vector<U, N> result;
    std::size_t i = 0;
    ((result[i++] = static_cast<U>(args)), ...);
    return result;
}

constexpr std::size_t CLI_DIM = 3;
using CliVector = Vector<double, CLI_DIM>;

void printMenu() {
    std::cout << "\n=== Меню операцій над векторами ===\n";
    std::cout << "1. Ввести вектори\n";
    std::cout << "2. Додати вектори\n";
    std::cout << "3. Відняти вектори\n";
    std::cout << "4. Множення вектора на скаляр\n";
    std::cout << "5. Ділення вектора на скаляр\n";
    std::cout << "6. Вивести поточні вектори\n";
    std::cout << "0. Вийти\n";
    std::cout << "Оберіть опцію: ";
}

void inputVector(CliVector &v, const std::string &name) {
    std::cout << "Введіть " << name << " (" << CLI_DIM << " значень): ";
    for (std::size_t i = 0; i < CLI_DIM; ++i) {
        std::cin >> v[i];
    }
}

int main() {
    CliVector v1, v2;
    bool hasInput = false;
    int choice;
    double scalar;

    while (true) {
        printMenu();
        std::cin >> choice;
        switch (choice) {
            case 1:
                inputVector(v1, "вектор 1");
                inputVector(v2, "вектор 2");
                hasInput = true;
                break;
            case 2:
                if (!hasInput) { std::cout << "Будь ласка, спочатку введіть вектори!\n"; break; }
                std::cout << "v1 + v2 = " << (v1 + v2) << "\n";
                break;
            case 3:
                if (!hasInput) { std::cout << "Будь ласка, спочатку введіть вектори!\n"; break; }
                std::cout << "v1 - v2 = " << (v1 - v2) << "\n";
                break;
            case 4:
                if (!hasInput) { std::cout << "Будь ласка, спочатку введіть вектори!\n"; break; }
                std::cout << "Введіть скаляр: ";
                std::cin >> scalar;
                std::cout << "v1 * скаляр = " << (v1 * scalar) << "\n";
                std::cout << "v2 * скаляр = " << (v2 * scalar) << "\n";
                break;
            case 5:
                if (!hasInput) { std::cout << "Будь ласка, спочатку введіть вектори!\n"; break; }
                std::cout << "Введіть скаляр: ";
                std::cin >> scalar;
                try {
                    std::cout << "v1 / скаляр = " << (v1 / scalar) << "\n";
                    std::cout << "v2 / скаляр = " << (v2 / scalar) << "\n";
                } catch (const std::exception &e) {
                    std::cout << "Помилка: " << e.what() << "\n";
                }
                break;
            case 6:
                if (!hasInput) { std::cout << "Будь ласка, спочатку введіть вектори!\n"; break; }
                std::cout << "Вектор 1: " << v1 << "\n";
                std::cout << "Вектор 2: " << v2 << "\n";
                break;
            case 0:
                std::cout << "Вихід. До побачення!\n";
                return 0;
            default:
                std::cout << "Неправильна опція, спробуйте ще раз.\n";
        }
    }

    return 0;
}
