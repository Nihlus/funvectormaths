#ifndef VEC_HPP_INCLUDED
#define VEC_HPP_INCLUDED

#include <math.h>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <float.h>
#include <random>
//#include <functional>

#define M_PI		3.14159265358979323846
#define M_PIf ((float)M_PI)

///bad, only for temporary debugging
#define EXPAND_3(vec) vec.v[0], vec.v[1], vec.v[2]
#define EXPAND_2(vec) vec.v[0], vec.v[1]

template<int N, typename T>
struct vec
{
    T v[N];

    vec(std::initializer_list<T> init)
    {
        if(init.size() == 1)
        {
            for(int i=0; i<N; i++)
            {
                v[i] = *(init.begin());
            }
        }
        else
        {
            int i;

            for(i=0; i<init.size(); i++)
            {
                v[i] = *(init.begin() + i);
            }

            for(; i<N; i++)
            {
                v[i] = 0.f;
            }
        }
    }

    T& x()
    {
        return v[0];
    }

    T& y()
    {
        return v[1];
    }

    T& z()
    {
        return v[2];
    }

    T& w()
    {
        return v[3];
    }

    vec<2, T>& xy()
    {
        ///umm. I think I might have committed some sort of deadly sin
        ///although I believe the C spec is slightly more permitting of this than
        ///you might believe at first sight
        vec<2, T>& ret = *(vec<2, T>*)&v[0];

        return ret;
    }

    vec<2, T>& yz()
    {
        ///umm. I think I might have committed some sort of deadly sin
        ///although I believe the C spec is slightly more permitting of this than
        ///you might believe at first sight
        vec<2, T>& ret = *(vec<2, T>*)&v[1];

        return ret;
    }

    ///can't figure out an easy way to make this referency
    vec<2, T> xz()
    {
        return {v[0], v[2]};
    }

    ///lets modernise the code a little
    vec()
    {
        for(int i=0; i<N; i++)
        {
            v[i] = T();
        }
    }

    vec(T val)
    {
        for(int i=0; i<N; i++)
        {
            v[i] = val;
        }
    }

    vec<N, T>& operator=(T val)
    {
        for(int i=0; i<N; i++)
        {
            v[i] = val;
        }

        return *this;
    }

    vec<N, T> operator+(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] + other.v[i];
        }

        return r;
    }

    vec<N, T> operator+(T other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] + other;
        }

        return r;
    }

    vec<N, T>& operator+=(T other)
    {
        *this = *this + other;

        return *this;
    }

    vec<N, T>& operator+=(const vec<N, T>& other)
    {
        *this = *this + other;

        return *this;
    }

    vec<N, T> operator-(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] - other.v[i];
        }

        return r;
    }


    vec<N, T> operator-(T other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] - other;
        }

        return r;
    }


    vec<N, T> operator*(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] * other.v[i];
        }

        return r;
    }

    ///beginnings of making this actually work properly
    template<typename U>
    vec<N, T> operator*(U other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] * other;
        }

        return r;
    }

    vec<N, T> operator/(const vec<N, T>& other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] / other.v[i];
        }

        return r;
    }

    vec<N, T> operator/(T other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] / other;
        }

        return r;
    }

    template<typename U>
    vec<N, U> map(std::function<U(T)> func)
    {
        vec<N, U> ret;

        for(int i=0; i<N; i++)
        {
            ret.v[i] = func(v[i]);
        }

        return ret;
    }

    inline
    T squared_length() const
    {
        T sqsum = 0;

        for(int i=0; i<N; i++)
        {
            sqsum += v[i]*v[i];
        }

        return sqsum;
    }

    inline
    T length() const
    {
        T l = squared_length();

        T val = sqrt(l);

        return val;
    }

    inline
    T lengthf() const
    {
        T l = squared_length();

        T val = sqrtf(l);

        return val;
    }

    double length_d() const
    {
        double l = 0;

        for(int i=0; i<N; i++)
        {
            l += v[i]*v[i];
        }

        return sqrt(l);
    }

    float sum() const
    {
        float accum = 0;

        for(int i=0; i<N; i++)
        {
            accum += v[i];
        }

        return accum;
    }

    float sum_absolute() const
    {
        float accum = 0;

        for(int i=0; i<N; i++)
        {
            accum += fabs(v[i]);
        }

        return accum;
    }

    float max_elem() const
    {
        float val = -FLT_MAX;

        for(const auto& s : v)
        {
            if(s > val)
                val = s;
        }

        return val;
    }

    float min_elem() const
    {
        float val = FLT_MAX;

        for(const auto& s : v)
        {
            if(s < val)
                val = s;
        }

        return val;
    }

    int which_element_minimum() const
    {
        float val = FLT_MAX;
        int num = -1;

        for(int i=0; i<N; i++)
        {
            if(v[i] < val)
            {
                val = v[i];
                num = i;
            }
        }

        return num;
    }

    float largest_elem() const
    {
        float val = -1;

        for(const auto& s : v)
        {
            float r = fabs(s);

            if(r > val)
                val = r;
        }

        return val;
    }


    vec<N, T> norm() const
    {
        T len = length();

        if(len < 0.00001f)
        {
            vec<N, T> ret;

            for(int i=0; i<N; i++)
                ret.v[i] = 0.f;

            return ret;
        }

        return (*this) / len;
    }

    ///only makes sense for a vec3f
    ///swap to matrices once I have any idea what is life
    ///glm or myself (;_;)
    vec<N, T> rot(const vec<3, float>& pos, const vec<3, float>& rotation) const
    {
        vec<3, float> c;
        vec<3, float> s;

        for(int i=0; i<3; i++)
        {
            c.v[i] = cos(rotation.v[i]);
            s.v[i] = sin(rotation.v[i]);
        }

        vec<3, float> rel = *this - pos;

        vec<3, float> ret;

        ret.v[0] = c.v[1] * (s.v[2] * rel.v[1] + c.v[2]*rel.v[0]) - s.v[1]*rel.v[2];
        ret.v[1] = s.v[0] * (c.v[1] * rel.v[2] + s.v[1]*(s.v[2]*rel.v[1] + c.v[2]*rel.v[0])) + c.v[0]*(c.v[2]*rel.v[1] - s.v[2]*rel.v[0]);
        ret.v[2] = c.v[0] * (c.v[1] * rel.v[2] + s.v[1]*(s.v[2]*rel.v[1] + c.v[2]*rel.v[0])) - s.v[0]*(c.v[2]*rel.v[1] - s.v[2]*rel.v[0]);

        return ret;
    }

    vec<N, T> back_rot(const vec<3, float>& position, const vec<3, float>& rotation) const
    {
        /*vec<3, float> new_pos = this->rot(position, (vec<3, float>){-rotation.v[0], 0, 0});
        new_pos = new_pos.rot(position, (vec<3, float>){0, -rotation.v[1], 0});
        new_pos = new_pos.rot(position, (vec<3, float>){0, 0, -rotation.v[2]});

        return new_pos;*/

        vec<N, T> c, s;

        for(int i=0; i<N; i++)
        {
            c.v[i] = cos(rotation.v[i]);
            s.v[i] = sin(rotation.v[i]);
        }

        vec<N, T> rel = *this - position;

        vec<N, T> ret;

        ret.x() = c.y() * c.z() * rel.x() + (s.x() * s.y() * c.z() - c.x() * s.z()) * rel.y() + (c.x() * s.y() * c.z() + s.x() * s.z()) * rel.z();
        ret.y() = (s.z() * c.y()) * rel.x() + (c.x() * c.z() + s.x() * s.y() * s.z()) * rel.y() + (-s.x() * c.z() + c.x() * s.y() * s.z()) * rel.z();
        ret.z() = -s.y() * rel.x() + (s.x() * c.y()) * rel.y() + (c.x() * c.y()) * rel.z();

        return ret;
    }

    ///only valid for a 2-vec
    ///need to rejiggle the templates to work this out
    vec<2, T> rot(T rot_angle)
    {
        T len = length();

        if(len < 0.00001f)
            return *this;

        T cur_angle = angle();

        T new_angle = cur_angle + rot_angle;

        T nx = len * cos(new_angle);
        T ny = len * sin(new_angle);

        return {nx, ny};
    }

    T angle()
    {
        return atan2(v[1], v[0]);
    }

    ///from top
    vec<3, T> get_euler() const
    {
        static_assert(N == 3, "Can only convert 3 element vectors into euler angles");

        vec<3, T> dir = *this;

        float cangle = dot((vec<3, T>){0, 1, 0}, dir.norm());

        float angle2 = acos(cangle);

        float y = atan2(dir.v[2], dir.v[0]);

        ///z y x then?
        vec<3, T> rot = {0, y, angle2};

        return rot;
    }

    ///so y is along rectangle axis
    ///remember, z is not along the rectangle axis, its perpendicular to x
    ///extrinsic x, y, z
    ///intrinsic z, y, x

    /*vec<3, T> get_euler_alt() const
    {
        vec<3, T> dir = *this;

        vec<3, T> pole = {0, 1, 0};

        float angle_to_pole = -atan2(dir.v[2], dir.v[1]);//acos(dot(pole, dir.norm()));

        float zc = angle_to_pole;//M_PI/2.f - angle_to_pole;

        float xc = atan2(dir.v[0], dir.v[1]) + M_PI;

        //static float test = 0.f;
        //test += 0.001f;

        vec<3, T> rot = {zc, 0.f, -xc};

        return rot;
    }*/

    operator float() const
    {
        static_assert(N == 1, "Implicit float can conversion only be used on vec<1,T> types");

        return v[0];
    }

    friend std::ostream& operator<<(std::ostream& os, const vec<N, T>& v1)
    {
        for(int i=0; i<N-1; i++)
        {
            os << std::to_string(v1.v[i]) << " ";
        }

        os << std::to_string(v1.v[N-1]);

        return os;
    }
};

template<int N, typename T>
inline
vec<2, T> s_xz(const vec<N, T>& v)
{
    static_assert(N >= 3, "Not enough elements for xz swizzle");

    return {v.v[0], v.v[2]};
}

template<int N, typename T>
inline
vec<2, T> s_xy(const vec<N, T>& v)
{
    static_assert(N >= 2, "Not enough elements for xy swizzle");

    return {v.v[0], v.v[1]};
}

template<int N, typename T>
inline
vec<2, T> s_yz(const vec<N, T>& v)
{
    static_assert(N >= 3, "Not enough elements for xy swizzle");

    return {v.v[1], v.v[2]};
}

template<int N, typename T>
inline
vec<2, T> s_zy(const vec<N, T>& v)
{
    static_assert(N >= 3, "Not enough elements for xy swizzle");

    return {v.v[2], v.v[1]};
}

template<int N, typename T>
inline
vec<3, T> s_xz_to_xyz(const vec<2, T>& v)
{
    return {v.v[0], 0.f, v.v[1]};
}

template<int N, typename T>
inline
vec<N, T> sqrtf(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = sqrtf(v.v[i]);
    }

    return ret;
}

template<typename T, typename U>
T approx_equal(T v1, U v2)
{
    float bound = 0.0001f;

    return v1 >= v2 - bound && v1 < v2 + bound;
}

///r theta, phi
template<typename T>
inline
vec<3, T> cartesian_to_polar(const vec<3, T>& cartesian)
{
    float r = cartesian.length();
    float theta = acos(cartesian.v[2] / r);
    float phi = atan2(cartesian.v[1], cartesian.v[0]);

    return {r, theta, phi};
}

template<typename T>
inline
vec<3, T> polar_to_cartesian(const vec<3, T>& polar)
{
    float r = polar.v[0];
    float theta = polar.v[1];
    float phi = polar.v[2];

    float x = r * sin(theta) * cos(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(theta);

    return {x, y, z};
}

template<typename T>
inline
vec<2, T> radius_angle_to_vec(T radius, T angle)
{
    vec<2, T> ret;

    ret.v[0] = cos(angle) * radius;
    ret.v[1] = sin(angle) * radius;

    return ret;
}

template<int N, typename T>
inline
vec<N, T> round(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = roundf(v.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> round_to_multiple(const vec<N, T>& v, int multiple)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v.v[i] / multiple;
        ret.v[i] = round(ret.v[i]);
        ret.v[i] *= multiple;
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> vcos(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = cos(v.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> vsin(const vec<N, T>& v)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = sin(v.v[i]);
    }

    return ret;
}

/*template<int N, typename T>
bool operator<(const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
    {
        if(v1.v[i] < v2.v[i])
            return true;
        if(v1.v[i] > v2.v[i])
            return false;
    }

    return false;
}*/

template<int N, typename T>
bool operator<(const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
        if(v1.v[i] >= v2.v[i])
            return false;

    return true;
}

template<int N, typename T>
bool operator>(const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
        if(v1.v[i] <= v2.v[i])
            return false;

    return true;
}

template<int N, typename T>
bool operator== (const vec<N, T>& v1, const vec<N, T>& v2)
{
    for(int i=0; i<N; i++)
        if(v1.v[i] != v2.v[i])
            return false;

    return true;
}

template<int N, typename T>
bool operator>= (const vec<N, T>& v1, const vec<N, T>& v2)
{
    return v1 > v2 || v1 == v2;
}

#define V3to4(x) {x.v[0], x.v[1], x.v[2], x.v[3]}

typedef vec<4, float> vec4f;
typedef vec<3, float> vec3f;
typedef vec<2, float> vec2f;

typedef vec<4, int> vec4i;
typedef vec<3, int> vec3i;
typedef vec<2, int> vec2i;

template<int N, typename T>
inline
vec<N, T> val_to_vec(float val)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = val;
    }

    return ret;
}
inline float randf_s()
{
    return (float)rand() / (RAND_MAX + 1.f);
}

///both of these functions are stolen shamelessly off stackoverflow
///https://stackoverflow.com/questions/686353/c-random-float-number-generation
inline float randf_s(float M, float N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;
}

template<int N, typename T>
bool any_nan(const vec<N, T>& v)
{
    for(int i=0; i<N; i++)
    {
        if(std::isnan(v.v[i]))
            return true;
    }

    return false;
}

template<int N, typename T>
inline
vec<N, T> randf(float M, float MN)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s(M, MN);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> randv(const vec<N, T>& vm, const vec<N, T>& vmn)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s(vm.v[i], vmn.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> randf()
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s();
    }

    return ret;
}

inline
float rand_det_s(std::minstd_rand& rnd, float M, float MN)
{
    float scaled = (rnd() - rnd.min()) / (float)(rnd.max() - rnd.min());

    return scaled * (MN - M) + M;
}

template<int N, typename T>
inline
vec<N, T> rand_det(std::minstd_rand& rnd, const vec<N, T>& M, const vec<N, T>& MN)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = rand_det_s(rnd, M.v[i], MN.v[i]);
    }

    return ret;
}

template<int N, typename T, typename U, typename V>
inline
vec<N, T> clamp(vec<N, T> v1, U p1, V p2)
{
    for(int i=0; i<N; i++)
    {
        v1.v[i] = v1.v[i] < p1 ? p1 : v1.v[i];
        v1.v[i] = v1.v[i] > p2 ? p2 : v1.v[i];
    }

    return v1;
}

template<typename T, typename U, typename V>
inline
T clamp(T v1, U p1, V p2)
{
    v1 = v1 < p1 ? p1 : v1;
    v1 = v1 > p2 ? p2 : v1;

    return v1;
}

template<int N, typename T, typename U, typename V>
inline
vec<N, T> clamp(const vec<N, T>& v1, const vec<N, U>& p1, const vec<N, V>& p2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v1.v[i] < p1.v[i] ? p1.v[i] : v1.v[i];
        ret.v[i] = ret.v[i] > p2.v[i] ? p2.v[i] : ret.v[i];
    }

    return ret;
}


///0 -> 1, returns packed RGBA uint
inline
uint32_t rgba_to_uint(const vec<4, float>& rgba)
{
    vec<4, float> val = clamp(rgba, 0.f, 1.f);

    uint8_t r = val.v[0] * 255;
    uint8_t g = val.v[1] * 255;
    uint8_t b = val.v[2] * 255;
    uint8_t a = val.v[3] * 255;

    uint32_t ret = (r << 24) | (g << 16) | (b << 8) | a;

    return ret;
}

inline
uint32_t rgba_to_uint(const vec<3, float>& rgb)
{
    return rgba_to_uint((vec4f){rgb.v[0], rgb.v[1], rgb.v[2], 1.f});
}

inline vec3f rot(const vec3f& p1, const vec3f& pos, const vec3f& rot)
{
    return p1.rot(pos, rot);
}

inline vec3f back_rot(const vec3f& p1, const vec3f& pos, const vec3f& rot)
{
    return p1.back_rot(pos, rot);
}

inline vec3f cross(const vec3f& v1, const vec3f& v2)
{
    vec3f ret;

    ret.v[0] = v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1];
    ret.v[1] = v1.v[2] * v2.v[0] - v1.v[0] * v2.v[2];
    ret.v[2] = v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0];

    return ret;
}

///counterclockwise
template<int N, typename T>
inline vec<N, T> perpendicular(const vec<N, T>& v1)
{
    return {-v1.v[1], v1.v[0]};
}

template<int N, typename T>
inline T dot(const vec<N, T>& v1, const vec<N, T>& v2)
{
    T ret = 0;

    for(int i=0; i<N; i++)
    {
        ret += v1.v[i] * v2.v[i];
    }

    return ret;
}

template<int N, typename T>
inline
T angle_between_vectors(const vec<N, T>& v1, const vec<N, T>& v2)
{
    return acos(clamp(dot(v1.norm(), v2.norm()), -1.f, 1.f));
}

template<int N, typename T>
inline vec<N, T> operator-(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = -v1.v[i];
    }

    return ret;
}

template<int N, typename T, typename U>
inline vec<N, U> operator*(T v, const vec<N, U>& v1)
{
    return v1 * v;
}

template<int N, typename T>
inline vec<N, T> operator+(T v, const vec<N, T>& v1)
{
    return v1 + v;
}

/*inline vec3f operator-(float v, const vec3f& v1)
{
    return v1 - v;
}*/

///should convert these functions to be N/T

template<int N, typename T>
inline vec<N, T> operator/(T v, const vec<N, T>& v1)
{
    vec<N, T> top;

    for(int i=0; i<N; i++)
        top.v[i] = v;

    return top / v1;
}

inline float r2d(float v)
{
    return (v / (M_PI*2.f)) * 360.f;
}

template<int N, typename T>
inline vec<N, T> fabs(const vec<N, T>& v)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = fabs(v.v[i]);
    }

    return v1;
}

template<int N, typename T, typename U>
inline vec<N, T> fmod(const vec<N, T>& v, U n)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = fmod(v.v[i], n);
    }

    return v1;
}

template<typename T, typename U>
inline T modulus_positive(T t, U n)
{
    ///floor not trunc like fmod
    return t - floor(t/n) * n;
}

template<int N, typename T, typename U>
inline vec<N, T> modulus_positive(const vec<N, T>& v, U n)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = modulus_positive(v.v[i], n);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> frac(const vec<N, T>& v)
{
    vec<N, T> v1 = fmod(v, (T)1.);

    return v1;
}

template<int N, typename T>
inline vec<N, T> floor(const vec<N, T>& v)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = floorf(v.v[i]);
    }

    return v1;
}

template<int N, typename T>
inline vec<N, T> ceil(const vec<N, T>& v)
{
    vec<N, T> v1;

    for(int i=0; i<N; i++)
    {
        v1.v[i] = ceilf(v.v[i]);
    }

    return v1;
}

template<int N, typename T>
inline vec<N, T> ceil_away_from_zero(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        if(v1.v[i] >= 0)
            ret.v[i] = ceil(v1.v[i]);
        if(v1.v[i] < 0)
            ret.v[i] = floor(v1.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> min(const vec<N, T>& v1, const vec<N, T>& v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::min(v1.v[i], v2.v[i]);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> max(const vec<N, T>& v1, const vec<N, T>& v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::max(v1.v[i], v2.v[i]);
    }

    return ret;
}

template<int N, typename T, typename U>
inline vec<N, T> min(const vec<N, T>& v1, U v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::min(v1.v[i], (T)v2);
    }

    return ret;
}

template<int N, typename T, typename U>
inline vec<N, T> max(const vec<N, T>& v1, U v2)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = std::max(v1.v[i], (T)v2);
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> axis_angle(const vec<N, T>& v, const vec<N, T>& axis, float angle)
{
    return cos(angle) * v + sin(angle) * cross(axis, v) + (1.f - cos(angle)) * dot(axis, v) * axis;
}


inline
vec3f aa_to_euler(const vec3f& axis, float angle)
{
    vec3f naxis = axis.norm();

    float s = sin(angle);
	float c = cos(angle);
	float t = 1-c;

	float x = naxis.v[0];
	float y = naxis.v[0];
	float z = naxis.v[0];

	if ((x*y*t + z*s) > 0.998f)
    { // north pole singularity detected
		float heading = 2*atan2(x*sin(angle/2), cos(angle/2));
		float attitude = M_PI/2;
		float bank = 0;

		return {attitude, heading, bank};
	}

	if ((x*y*t + z*s) < -0.998)
    { // south pole singularity detected
		float heading = -2*atan2(x*sin(angle/2), cos(angle/2));
		float attitude = -M_PI/2;
		float bank = 0;

        return {attitude, heading, bank};
	}

	float heading = atan2(y * s- x * z * t , 1 - (y*y+ z*z ) * t);
	float attitude = asin(x * y * t + z * s) ;
	float bank = atan2(x * s - y * z * t , 1 - (x*x + z*z) * t);

	return {attitude, heading, bank};
}

template<typename U>
inline vec<4, float> rgba_to_vec(const U& rgba)
{
    vec<4, float> ret;

    ret.v[0] = rgba.r;
    ret.v[1] = rgba.g;
    ret.v[2] = rgba.b;
    ret.v[3] = rgba.a;

    return ret;
}

///could probably sfinae this
template<typename U>
inline vec<4, float> xyzw_to_vec(const U& xyzw)
{
    vec<4, float> ret;

    ret.v[0] = xyzw.x;
    ret.v[1] = xyzw.y;
    ret.v[2] = xyzw.z;
    ret.v[3] = xyzw.w;

    return ret;
}

///could probably sfinae this
template<typename U>
inline vec<4, float> xyzwf_to_vec(const U& xyzw)
{
    vec<4, float> ret;

    ret.v[0] = xyzw.x();
    ret.v[1] = xyzw.y();
    ret.v[2] = xyzw.z();
    ret.v[3] = xyzw.w();

    return ret;
}

///could probably sfinae this
template<typename U>
inline vec<3, float> xyz_to_vec(const U& xyz)
{
    vec<3, float> ret;

    ret.v[0] = xyz.x;
    ret.v[1] = xyz.y;
    ret.v[2] = xyz.z;

    return ret;
}

/*template<typename U, typename T = decltype(U::x())>
inline vec<3, float> xyz_to_vec(const U& xyz)
{
    vec<3, float> ret;

    ret.v[0] = xyz.x();
    ret.v[1] = xyz.y();
    ret.v[2] = xyz.z();

    return ret;
}*/

template<typename U>
inline vec<3, float> xyzf_to_vec(const U& xyz)
{
    vec<3, float> ret;

    ret.v[0] = xyz.x();
    ret.v[1] = xyz.y();
    ret.v[2] = xyz.z();

    return ret;
}

template<typename U>
inline vec<2, float> xy_to_vec(const U& xyz)
{
    vec<2, float> ret;

    ret.v[0] = xyz.x;
    ret.v[1] = xyz.y;

    return ret;
}

template<typename U, typename T, int N>
inline vec<N, T> conv(const vec<N, U>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v1.v[i];
    }

    return ret;
}

///there's probably a better way of doing this
template<typename U, typename T>
inline U conv_implicit(const vec<2, T>& v1)
{
    return {v1.v[0], v1.v[1]};
}

template<typename U, typename T>
inline U conv_implicit(const vec<3, T>& v1)
{
    return {v1.v[0], v1.v[1], v1.v[2]};
}

template<typename U, typename T>
inline U conv_implicit(const vec<4, T>& v1)
{
    return {v1.v[0], v1.v[1], v1.v[2], v1.v[3]};
}

template<typename U, typename T>
inline U conv_implicit(const T& q)
{
    return {q.q.v[0], q.q.v[1], q.q.v[2], q.q.v[3]};
}

template<int N, typename T>
inline vec<N, T> d2r(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = (v1.v[i] / 360.f) * M_PI * 2;
    }

    return ret;
}

inline float mix(float p1, float p2, float a)
{
    return p1 * (1.f - a) + p2 * a;
}

template<int N, typename T>
inline vec<N, T> mix(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = v1.v[i] * (1.f - a) + v2.v[i] * a;//v1.v[i] + (v2.v[i] - v1.v[i]) * a;
    }

    return ret;
}

template<int N, typename T>
inline vec<N, T> mix3(const vec<N, T>& v1, const vec<N, T>& mid, const vec<N, T>& v2, float a)
{
    if(a <= 0.5f)
    {
        return mix(v1, mid, a * 2.f);
    }
    else
    {
        return mix(mid, v2, (a - 0.5f) * 2.f);
    }
}

/*template<int N, typename T>
inline vec<N, T> slerp(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    vec<N, T> ret;

    ///im sure you can convert the cos of a number to the sign, rather than doing this
    float angle = acos(dot(v1.norm(), v2.norm()));

    if(angle < 0.0001f && angle >= -0.0001f)
        return mix(v1, v2, a);

    float a1 = sin((1 - a) * angle) / sin(angle);
    float a2 = sin(a * angle) / sin(angle);

    ret = a1 * v1 + a2 * v2;

    return ret;
}*/

template<int N, typename T>
inline vec<N, T> slerp(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    vec<N, T> ret;

    ///im sure you can convert the cos of a number to the sign, rather than doing this
    float angle = acos(dot(v1, v2) / (v1.length() * v2.length()));

    if(angle < 0.00001f && angle >= -0.00001f)
        return mix(v1, v2, a);

    float a1 = sin((1 - a) * angle) / sin(angle);
    float a2 = sin(a * angle) / sin(angle);

    ret = a1 * v1 + a2 * v2;

    return ret;
}

template<int N, typename T>
inline vec<N, T> slerp3(const vec<N, T>& v1, const vec<N, T>& mid, const vec<N, T>& v2, float a)
{
    if(a <= 0.5f)
    {
        return slerp(v1, mid, a * 2.f);
    }
    else
    {
        return slerp(mid, v2, (a - 0.5f) * 2.f);
    }
}

template<int N, typename T>
inline
vec<N, T> cosint(const vec<N, T>& v1, const vec<N, T>& v2, float a)
{
    float mu2 = (1.f-cosf(a*M_PI))/2.f;

    return v1*(1-mu2) + v2*mu2;
}

template<int N, typename T>
inline
vec<N, T> cosint3(const vec<N, T>& v1, const vec<N, T>& mid, const vec<N, T>& v2, float a)
{
    if(a <= 0.5f)
    {
        return cosint(v1, mid, a * 2.f);
    }
    else
    {
        return cosint(mid, v2, (a - 0.5f) * 2.f);
    }
}

/*template<int N, typename T>
inline
vec<N, T> rejection(const vec<N, T>& v1, const vec<N, T>& v2)
{
    /*vec<N, T> me_to_them = v2 - v1;

    me_to_them = me_to_them.norm();

    float scalar_proj = dot(move_dir, me_to_them);

    vec<N, T> to_them_relative = scalar_proj * me_to_them;

    vec<N, T> perp = move_dir - to_them_relative;

    return perp;
}*/

/*float cosif3(float y1, float y2, float y3, float frac)
{
    float fsin = sin(frac * M_PI);

    float val;

    if(frac < 0.5f)
    {
        val = (1.f - fsin) * y1 + fsin * y2;
    }
    else
    {
        val = (fsin) * y2 + (1.f - fsin) * y3;
    }

    return val;
}*/

template<int N, typename T>
inline
vec<N, T> projection(const vec<N, T>& v1, const vec<N, T>& dir)
{
    vec<N, T> ndir = dir.norm();

    float a1 = dot(v1, ndir);

    return a1 * ndir;
}

inline vec3f generate_flat_normal(const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
    return cross(p2 - p1, p3 - p1).norm();
}

///t should be some container of vec3f
///sorted via 0 -> about vector, plane perpendicular to that
///I should probably make the second version an overload/specialisation, rather than a runtime check
///performance isnt 100% important though currently
template<typename T>
inline std::vector<vec3f> sort_anticlockwise(const T& in, const vec3f& about, std::vector<std::pair<float, int>>* pass_out = nullptr)
{
    int num = in.size();

    std::vector<vec3f> out;
    std::vector<std::pair<float, int>> intermediate;

    out.reserve(num);
    intermediate.reserve(num);

    vec3f euler = about.get_euler();

    vec3f centre_point = about.back_rot(0.f, euler);

    vec2f centre_2d = (vec2f){centre_point.v[0], centre_point.v[2]};

    for(int i=0; i<num; i++)
    {
        vec3f vec_pos = in[i];

        vec3f rotated = vec_pos.back_rot(0.f, euler);

        vec2f rot_2d = (vec2f){rotated.v[0], rotated.v[2]};

        vec2f rel = rot_2d - centre_2d;

        float angle = rel.angle();

        intermediate.push_back({angle, i});
    }

    std::sort(intermediate.begin(), intermediate.end(),
              [](auto i1, auto i2)
              {
                  return i1.first < i2.first;
              }
              );

    for(auto& i : intermediate)
    {
        out.push_back(in[i.second]);
    }

    if(pass_out != nullptr)
    {
        *pass_out = std::move(intermediate);
    }

    return out;
}

template<int N, typename T, typename U>
inline
void line_draw_helper(const vec<N, T>& start, const vec<N, T>& finish, vec<N, T>& out_dir, U& num)
{
    vec<N, T> dir = (finish - start);
    T dist = dir.largest_elem();

    dir = dir / dist;

    out_dir = dir;
    num = dist;
}

///there is almost certainly a better way to do this
///this function doesn't work properly, don't use it
inline
float circle_minimum_distance(float v1, float v2)
{
    v1 = fmodf(v1, M_PI * 2.f);
    v2 = fmodf(v2, M_PI * 2.f);

    float d1 = fabs(v2 - v1);
    float d2 = fabs(v2 - v1 - M_PI*2.f);
    float d3 = fabs(v2 - v1 + M_PI*2.f);

    vec3f v = (vec3f){d1, d2, d3};

    if(v.which_element_minimum() == 0)
    {
        return v2 - v1;
    }

    if(v.which_element_minimum() == 1)
    {
        return v2 - v1 - M_PI*2.f;
    }

    if(v.which_element_minimum() == 2)
    {
        return v2 - v1 + M_PI*2.f;
    }

    //float result = std::min(d1, std::min(d2, d3));
}

///rename this function
template<int N, typename T>
inline
vec<N, T> point2line_shortest(const vec<N, T>& lp, const vec<N, T>& ldir, const vec<N, T>& p)
{
    vec<N, T> ret;

    auto n = ldir.norm();

    ret = (lp - p) - dot(lp - p, n) * n;

    return ret;
}

///https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
template<typename T>
inline
bool is_left_side(const vec<2, T>& l1, const vec<2, float>& l2, const vec<2, float>& lp)
{
    return ((l2.v[0] - l1.v[0]) * (lp.v[1] - l1.v[1]) - (l2.v[1] - l1.v[1]) * (lp.v[0] - l1.v[0])) > 0;
}

///nicked off stackoverflow
template <typename T> int signum(T val) {
    return (T(0) < val) - (val < T(0));
}

///don't want to have to include all of algorithm for just this function
template<class T>
const T& max(const T& a, const T& b)
{
    return (a < b) ? b : a;
}

template<int N, typename T>
struct mat
{
    T v[N][N] = {{0}};

    mat()
    {
        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
            {
                v[j][i] = 0;
            }
        }
    }

    mat<N, T> from_vec(vec3f v1, vec3f v2, vec3f v3) const
    {
        mat<N, T> m;

        for(int i=0; i<3; i++)
            m.v[0][i] = v1.v[i];

        for(int i=0; i<3; i++)
            m.v[1][i] = v2.v[i];

        for(int i=0; i<3; i++)
            m.v[2][i] = v3.v[i];

        return m;
    }

    void load(vec3f v1, vec3f v2, vec3f v3)
    {
        for(int i=0; i<3; i++)
            v[0][i] = v1.v[i];

        for(int i=0; i<3; i++)
            v[1][i] = v2.v[i];

        for(int i=0; i<3; i++)
            v[2][i] = v3.v[i];
    }

    float det() const
    {
        static_assert(N == 3, "N must be 3");

        float a11, a12, a13, a21, a22, a23, a31, a32, a33;

        a11 = v[0][0];
        a12 = v[0][1];
        a13 = v[0][2];

        a21 = v[1][0];
        a22 = v[1][1];
        a23 = v[1][2];

        a31 = v[2][0];
        a32 = v[2][1];
        a33 = v[2][2];

        ///get determinant
        float d = a11*a22*a33 + a21*a32*a13 + a31*a12*a23 - a11*a32*a23 - a31*a22*a13 - a21*a12*a33;

        return d;
    }

    mat<3, T> invert() const
    {
        float d = det();

        float a11, a12, a13, a21, a22, a23, a31, a32, a33;

        a11 = v[0][0];
        a12 = v[0][1];
        a13 = v[0][2];

        a21 = v[1][0];
        a22 = v[1][1];
        a23 = v[1][2];

        a31 = v[2][0];
        a32 = v[2][1];
        a33 = v[2][2];


        vec3f ir1, ir2, ir3;

        ir1.v[0] = a22 * a33 - a23 * a32;
        ir1.v[1] = a13 * a32 - a12 * a33;
        ir1.v[2] = a12 * a23 - a13 * a22;

        ir2.v[0] = a23 * a31 - a21 * a33;
        ir2.v[1] = a11 * a33 - a13 * a31;
        ir2.v[2] = a13 * a21 - a11 * a23;

        ir3.v[0] = a21 * a32 - a22 * a31;
        ir3.v[1] = a12 * a31 - a11 * a32;
        ir3.v[2] = a11 * a22 - a12 * a21;

        ir1 = ir1 * d;
        ir2 = ir2 * d;
        ir3 = ir3 * d;

        return from_vec(ir1, ir2, ir3);
    }

    vec<3, T> get_v1() const
    {
        return {v[0][0], v[0][1], v[0][2]};
    }
    vec<3, T> get_v2() const
    {
        return {v[1][0], v[1][1], v[1][2]};
    }
    vec<3, T> get_v3() const
    {
        return {v[2][0], v[2][1], v[2][2]};
    }

    mat<N, T> identity()
    {
        mat<N, T> ret;

        for(int i=0; i<N; i++)
        {
            ret.v[i][i] = 1;
        }

        return ret;
    }

    ///ffs this was wrong
    mat<3, float> skew_symmetric_cross_product(vec3f cr)
    {
        mat<3, float> ret;

        ret.load({0, -cr.v[2], cr.v[1]},
                 {cr.v[2], 0, -cr.v[0]},
                 {-cr.v[1], cr.v[0], 0});

        return ret;
    }

    #if 0
    void from_dir(vec3f dir)
    {
        /*vec3f up = {0, 1, 0};

        vec3f xaxis = cross(up, dir).norm();
        vec3f yaxis = cross(dir, xaxis).norm();

        v[0][0] = xaxis.v[0];
        v[0][1] = yaxis.v[0];
        v[0][2] = dir.v[0];

        v[1][0] = xaxis.v[1];
        v[1][1] = yaxis.v[1];
        v[1][2] = dir.v[1];

        v[2][0] = xaxis.v[2];
        v[2][1] = yaxis.v[2];
        v[2][2] = dir.v[2];*/


    }
    #endif

    mat<3, float> XRot(float angle)
    {
        float c = cos(angle);
        float s = sin(angle);

        mat<3, float> v1;

        v1.load({1, 0, 0}, {0, c, s}, {0, -s, c});

        return v1;
    }

    mat<3, float> YRot(float angle)
    {
        float c = cos(angle);
        float s = sin(angle);

        mat<3, float> v2;

        v2.load({c, 0, -s}, {0, 1, 0}, {s, 0, c});

        return v2;
    }

    mat<3, float> ZRot(float angle)
    {
        float c = cos(angle);
        float s = sin(angle);

        mat<3, float> v3;

        v3.load({c, s, 0}, {-s, c, 0}, {0, 0, 1});

        return v3;
    }

    void load_rotation_matrix(vec3f _rotation)
    {
        /*vec3f c;
        vec3f s;

        for(int i=0; i<3; i++)
        {
            c.v[i] = cos(rotation.v[i]);
            s.v[i] = sin(rotation.v[i]);
        }

        ///rotation matrix
        //vec3f r1 = {c.v[1]*c.v[2], -c.v[1]*s.v[2], s.v[1]};
        //vec3f r2 = {c.v[0]*s.v[2] + c.v[2]*s.v[0]*s.v[1], c.v[0]*c.v[2] - s.v[0]*s.v[1]*s.v[2], -c.v[1]*s.v[0]};
        //vec3f r3 = {s.v[0]*s.v[2] - c.v[0]*c.v[2]*s.v[1], c.v[2]*s.v[0] + c.v[0]*s.v[1]*s.v[2], c.v[1]*c.v[0]};

        vec3f r1 = {c.v[1] * c.v[2], c.v[0] * s.v[2] + s.v[0] * s.v[1] * c.v[2], s.v[0] * s.v[2] - c.v[0] * s.v[1] * c.v[2]};
        vec3f r2 = {-c.v[1] * s.v[2], c.v[0] * c.v[2] - s.v[0] * s.v[1] * s.v[2], s.v[0] * c.v[2] + c.v[0] * s.v[1] * s.v[2]};
        vec3f r3 = {s.v[1], -s.v[0] * c.v[1], c.v[0] * c.v[1]};

        load(r1, r2, r3);*/

        vec3f rotation = {_rotation.v[0], _rotation.v[1], _rotation.v[2]};

        //rotation = rotation + M_PI/2.f;

        vec3f c = vcos(rotation);
        vec3f s = vsin(rotation);

        //printf("%f %f\n", c.v[1], s.v[1]);

        mat<3, float> v1;

        v1.load({1, 0, 0}, {0, c.v[0], s.v[0]}, {0, -s.v[0], c.v[0]});

        mat<3, float> v2;

        v2.load({c.v[1], 0, -s.v[1]}, {0, 1, 0}, {s.v[1], 0, c.v[1]});

        mat<3, float> v3;

        v3.load({c.v[2], s.v[2], 0}, {-s.v[2], c.v[2], 0}, {0, 0, 1});


        *this = v1 * v2 * v3;
    }

    ///fix the nan
    ///nanananananananana batman
    vec3f get_rotation()
    {
        vec3f rotation;

        ///need to deal with v[2][2] very close to 0

        float s2 = -v[0][2];

        ///so... s2 might be broken if we're here?
        bool ruh_roh = s2 < -0.995 || s2 >= 0.995;

        float possible_t1_1 = asin(s2);
        //float possible_t1_2 = M_PI - asin(s2);

        float cp1 = cos(possible_t1_1);
        //float cp2 = cos(possible_t1_2);

        ///should really handle the 0/0 / 0/0 case
        float possible_v0_1 = atan2(v[1][2], v[2][2]);
        //float possible_v0_2 = atan2(v[1][2] / cp2, v[2][2] / cp2);

        float possible_v2_1 = atan2(v[0][1], v[0][0]);
        //float possible_v2_2 = atan2(v[0][1] / cp2, v[0][0] / cp2);

        ///they both represent the same rotation, we just need to convert when we cross the midpoint
        //vec3f rotated_point = (vec3f){0, 0, 1}.rot({0,0,0}, {possible_v0_1, possible_t1_1, possible_v2_1});

        //vec3f normal = {0, 0, 1};

        //float cangle = dot(normal, rotated_point);

        ///nope
        ///but the problem is just which one of these two we pick
        ///so we're getting close
        ///we just need to define which hemisphere we're on
        ///which can't be THAT hard

        ///ok, so this doesnt work for z rotation
        //if(fabs(acos(cangle)) < M_PI/2)
        {
            return {possible_v0_1, possible_t1_1, possible_v2_1};
        }

        //return {possible_v0_2, possible_t1_2, possible_v2_2};

        ///c2 approximately 0
        ///v[0] wrong

        //float c2 = cos(asin(s2));

        //float natural_sign = 1;

        //if(c2 < 0)
        //    natural_sign = -1;

        //printf("natsign %f %f\n", natural_sign, s2);

        //rotation.v[0] = atan2(v[1][2], v[2][2]);
        //rotation.v[1] = asin(s2);
        //rotation.v[2] = atan2(v[0][1], v[0][0]);

        //printf("r1 %f %f %f\n", rotation.v[0], rotation.v[1], rotation.v[2]);


        ///alternate calculation for v[0]
        ///SIGH, have to handle all the fucking polarity cases
        ///piece of shit eat more wanker euler
        /*if(ruh_roh)
        {
            //rotation.v[0] = atan2(-v[2][1], v[1][1]);

            ///in the case of this, rotation.v[2] is ALSO broken
            ///FOR FUCKS SAKE
            ///WHAT IS THIS




            float s3 = sin(rotation.v[2]);
            float c3 = cos(rotation.v[2]);

            if(s2 > 0.995)
            {
                float c3m1 = (v[1][1] + v[2][0]) / 2;

                float a3m1 = acos(c3m1);

                rotation.v[0] = rotation.v[2] - a3m1;
            }
            else
            {
                float s3p1 = -(v[1][0] + v[2][1]) / 2;

                float a3p1 = asin(s3p1);

                rotation.v[0] = a3p1 - rotation.v[2];
            }
        }*/

        //printf("%f %f\n", v[0][0], v[0][2]);

        //printf("r0 %f s2 %f\n", rotation.v[0], s2);

        //printf("err %f %f %i\n", s2, v[2][2], ruh_roh);

        //printf("[2][2] %f\n", v[2][2]);

        return rotation;
    }

    vec<N, T> operator*(const vec<N, T>& other) const
    {
        vec<N, T> val;

        /*val.v[0] = v[0][0] * other.v[0] + v[0][1] * other.v[1] + v[0][2] * other.v[2];
        val.v[1] = v[1][0] * other.v[0] + v[1][1] * other.v[1] + v[1][2] * other.v[2];
        val.v[2] = v[2][0] * other.v[0] + v[2][1] * other.v[1] + v[2][2] * other.v[2];*/

        for(int i=0; i<N; i++)
        {
            float accum = 0;

            for(int j=0; j<N; j++)
            {
                accum += v[i][j] * other.v[j];
            }

            val.v[i] = accum;
        }

        return val;
    }

    mat<N, T> operator*(const mat<N, T>& other) const
    {
        mat<N, T> ret;

        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
            {
                //float val = v[j][0] * other.v[0][i] + v[j][1] * other.v[1][i] + v[j][2] * other.v[2][i];

                float accum = 0;

                for(int k=0; k<N; k++)
                {
                    accum += v[j][k] * other.v[k][i];
                }

                ret.v[j][i] = accum;
            }
        }

        return ret;
    }

    mat<3, T> operator*(T other) const
    {
        mat<3, T> ret;

        for(int j=0; j<3; j++)
        {
            for(int i=0; i<3; i++)
            {
                ret.v[j][i] = v[j][i] * other;
            }
        }

        return ret;
    }

    mat<N, T> operator+(const mat<N, T>& other) const
    {
        mat<N, T> ret;

        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
            {
                ret.v[j][i] = v[j][i] + other.v[j][i];
            }
        }

        return ret;
    }

    mat<N, T> transp()
    {
        mat<N, T> ret;

        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
            {
                ret.v[j][i] = v[i][j];
            }
        }

        return ret;
    }

    friend std::ostream& operator<<(std::ostream& os, const mat<N, T>& v1)
    {
        for(int j=0; j<N; j++)
        {
            for(int i=0; i<N; i++)
                os << std::to_string(v1.v[j][i]) << " ";

            os << std::endl;
        }

        return os;
    }
};

typedef mat<3, float> mat3f;

inline
mat3f tensor_product(vec3f v1, vec3f v2)
{
    mat3f ret;

    for(int j=0; j<3; j++)
    {
        for(int i=0; i<3; i++)
        {
            ret.v[j][i] = v1.v[j] * v2.v[i];
        }
    }

    return ret;
}

///need to fix for a == b = I, and a == -b = -I
///although the latter does not seem valid
inline
vec3f mat_from_dir(vec3f d1, vec3f d2)
{
    d1 = d1.norm();
    d2 = d2.norm();

    vec3f d = cross(d1, d2).norm();
    float c = dot(d1, d2);

    float s = sin(acos(c));

    ///so, d, acos(c) would be axis angle

    mat<3, float> skew_symmetric;
    skew_symmetric = skew_symmetric.skew_symmetric_cross_product(d);

    mat<3, float> unit;

    unit = unit.identity();

    mat<3, float> ret = unit * c + skew_symmetric * s + tensor_product(d, d) * (1.f - c);

    //std::cout << d << std::endl;

    return ret.get_rotation();
}

inline
mat3f map_unit_a_to_b(vec3f a, vec3f b)
{
    mat3f I;
    I = I.identity();

    a = a.norm();
    b = b.norm();

    vec3f v = cross(a, b);

    mat3f skew;

    skew = skew.skew_symmetric_cross_product(v);

    float s = v.length();
    float c = dot(a,b);

    //printf("sc %f %f\n", s, c);

    //std::cout << skew << "\n skew\n";

    if(c > 0.99995)
    {
        mat3f none;
        none.load_rotation_matrix({0,0,0});

        return none;
    }

    mat3f ret = I + skew + skew*skew * ((1 - c) / (s*s));

    return ret;
}

///http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
inline
mat3f axis_angle_to_mat(vec3f axis, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.f - c;

    axis = axis.norm();

    float x = axis.v[0];
    float y = axis.v[1];
    float z = axis.v[2];

    mat3f m;

    m.v[0][0] = t*x*x + c;
    m.v[0][1] = t*x*y - z*s;
    m.v[0][2] = t*x*z + y*s;

    m.v[1][0] = t*x*y + z*s;
    m.v[1][1] = t*y*y + c;
    m.v[1][2] = t*y*z - x*s;

    m.v[2][0] = t*x*z - y*s;
    m.v[2][1] = t*y*z + x*s;
    m.v[2][2] = t*z*z + c;

    return m;
}

struct quaternion
{
    vec4f q = {0,0,0,1};

    /*quaternion()
    {
        q = {0,0,0,1};
    }*/

    void load_from_matrix(const mat3f& m)
    {
        vec4f l;

        float m00 = m.v[0][0];
        float m11 = m.v[1][1];
        float m22 = m.v[2][2];

        l.v[3] = sqrt( max( 0.f, 1 + m00 + m11 + m22 ) ) / 2;
        l.v[0] = sqrt( max( 0.f, 1 + m00 - m11 - m22 ) ) / 2;
        l.v[1] = sqrt( max( 0.f, 1 - m00 + m11 - m22 ) ) / 2;
        l.v[2] = sqrt( max( 0.f, 1 - m00 - m11 + m22 ) ) / 2;

        float m21 = m.v[2][1];
        float m12 = m.v[1][2];
        float m02 = m.v[0][2];
        float m20 = m.v[2][0];
        float m10 = m.v[1][0];
        float m01 = m.v[0][1];

        //int s1 = signum(m21 - m12);
        //int s2 = signum(m02 - m20);
        //int s3 = signum(m10 - m01);

        l.v[0] = std::copysign( l.v[0], m21 - m12 );
        l.v[1] = std::copysign( l.v[1], m02 - m20 );
        l.v[2] = std::copysign( l.v[2], m10 - m01 );

        q = l;
    }

    ///this * ret == q
    quaternion get_difference(quaternion q)
    {
        mat3f t = get_rotation_matrix();
        mat3f o = q.get_rotation_matrix();

        ///A*q1 = q2
        ///A = q2 * q1'

        ///q1 * A = q2
        ///A = q1'q2

        mat3f diff = t.transp() * o;

        quaternion ret;
        ret.load_from_matrix(diff);

        return ret;
    }

    void load_from_euler(vec3f _rot)
    {
        mat3f mat;
        mat.load_rotation_matrix(_rot);

        load_from_matrix(mat);
    }

    void from_vec(const vec4f& raw)
    {
        q = raw;
    }

    quaternion operator*(const quaternion& other) const
    {
        quaternion ret;

        ret.q.v[0] = q.v[3] * other.q.v[0] + q.v[0] * other.q.v[3] + q.v[1] * other.q.v[2] - q.v[2] * other.q.v[1];
        ret.q.v[1] = q.v[3] * other.q.v[1] + q.v[1] * other.q.v[3] + q.v[2] * other.q.v[0] - q.v[0] * other.q.v[2];
        ret.q.v[2] = q.v[3] * other.q.v[2] + q.v[2] * other.q.v[3] + q.v[0] * other.q.v[1] - q.v[1] * other.q.v[0];
        ret.q.v[3] = q.v[3] * other.q.v[3] - q.v[0] * other.q.v[0] - q.v[1] * other.q.v[1] - q.v[2] * other.q.v[2];

        return ret;
    }

    ///http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
    ///we could use nlerp, but then the page provides a slerp implementation
    ///soo.. uuuh.. Sorry page author
    ///although they use numerically instable interpolation for the numerically unstable case,
    ///so now i feel like i can gloat at least
    ///http://www.mrpt.org/tutorials/programming/maths-and-geometry/slerp-interpolation/
    ///seems more legit
    static
    quaternion slerp(const quaternion& q1, const quaternion& q2, float t)
    {
        float d = dot(q1.q, q2.q);

        const float threshold = 0.9995f;

        if(d > threshold)
        {
            quaternion nq;

            nq.q = q1.q * (1.f - t) + q2.q * t;

            ///can... can this even not be normalised?
            ///i'm starting to trust this code less
            nq.q = nq.q.norm();

            return nq;
        }

        d = clamp(d, -1.f, 1.f);

        bool is_negative = false;

        if(d < 0)
        {
            is_negative = true;
            d = -d;
        }

        float theta = acos(d);
        float sintheta = sin(theta);

        float A = sin((1.f - t) * theta) / sintheta;
        float B = sin(t * theta) / sintheta;

        if(is_negative)
        {
            return {A * q1.q - B * q2.q};
        }

        return {A * q1.q + B * q2.q};
    }

    ///http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    mat3f get_rotation_matrix()
    {
        mat3f m;

        float qx = q.v[0];
        float qy = q.v[1];
        float qz = q.v[2];
        float qw = q.v[3];

        m.v[0][0] = 1 - 2*qy*qy - 2*qz*qz;
        m.v[0][1] = 2*qx*qy - 2*qz*qw;
        m.v[0][2] = 2*qx*qz + 2*qy*qw;

        m.v[1][0] = 2*qx*qy + 2*qz*qw;
        m.v[1][1] = 1 - 2*qx*qx - 2*qz*qz;
        m.v[1][2] = 2*qy*qz - 2*qx*qw;

        m.v[2][0] = 2*qx*qz - 2*qy*qw;
        m.v[2][1] = 2*qy*qz + 2*qx*qw;
        m.v[2][2] = 1 - 2*qx*qx - 2*qy*qy;

        return m;
    }

    quaternion norm()
    {
        /*float w = q.v[3];

        quaternion ret;

        ret.q.v[0] = q.v[0] / w;
        ret.q.v[1] = q.v[1] / w;
        ret.q.v[2] = q.v[2] / w;
        ret.q.v[3] = 1;

        return ret;*/

        quaternion ret;

        ret.q = q.norm();

        return ret;
    }

    quaternion conjugate()
    {
        quaternion ret;

        ret.q = q;

        ret.q.v[0] = -ret.q.v[0];
        ret.q.v[1] = -ret.q.v[1];
        ret.q.v[2] = -ret.q.v[2];

        return ret;
    }

    quaternion inverse()
    {
        quaternion conj = conjugate();

        vec4f l = conj.q / (q.lengthf() * q.lengthf());

        quaternion q;
        q.q = l;

        return q;
    }

    vec4f to_axis_angle()
    {
        float qx = q.v[0];
        float qy = q.v[1];
        float qz = q.v[2];
        float qw = q.v[3];

        float angle = 2 * acos(qw);
        float x = qx / sqrt(1-qw*qw);
        float y = qy / sqrt(1-qw*qw);
        float z = qz / sqrt(1-qw*qw);

        if(qw >= 0.999f)
        {
            x = 1;
            y = 0;
            z = 0;
        }

        return {x, y, z, angle};
    }

    void load_from_axis_angle(vec4f aa)
    {
        q.v[0] = aa.v[0] * sin(aa.v[3]/2);
        q.v[1] = aa.v[1] * sin(aa.v[3]/2);
        q.v[2] = aa.v[2] * sin(aa.v[3]/2);
        q.v[3] = cos(aa.v[3]/2);

        q = q.norm();
    }

    float x()
    {
       return q.x();
    }

    float y()
    {
       return q.y();
    }

    float z()
    {
       return q.z();
    }

    float w()
    {
       return q.w();
    }

    quaternion identity()
    {
        return {{0, 0, 0, 1}};
    }
};


inline
quaternion convert_leap_quaternion(quaternion q)
{
    vec4f aa = q.to_axis_angle();

    aa.v[0] = -aa.v[0];
    aa.v[1] = -aa.v[1];

    q.load_from_axis_angle(aa);

    return q;
}


template<typename T>
inline
quaternion convert_from_leap_quaternion(T ql)
{
    quaternion q = {{ql.x, ql.y, ql.z, ql.w}};

    return convert_leap_quaternion(q);
}

template<typename T>
inline
quaternion convert_from_bullet_quaternion(T ql)
{
    quaternion q = {{ql.x(), ql.y(), ql.z(), ql.w()}};

    return q;
}

///something i found in their source, ???
inline
mat3f leapquat_to_mat(quaternion q)
{
    float d = q.q.length() * q.q.length();

    float s = 2.f / d;

    vec4f v = q.q;

    float xs = v.v[0] * s, ys = v.v[1] * s, zs = v.v[2] * s;
    float wx = v.v[3] * xs, wy = v.v[3] * ys, wz = v.v[3] * zs;
    float xx = v.v[0] * xs, xy = v.v[0] * ys, xz = v.v[0] * zs;
    float yy = v.v[1] * ys, yz = v.v[1] * zs, zz = v.v[2] * zs;

    vec3f r1 = {1.f - (yy + zz), xy + wz, xz - wy};
    vec3f r2 = {xy - wz, 1.f - (xx + zz), yz + wx};
    vec3f r3 = {xz + wy, yz - wx, 1.f - (xx + yy)};

    mat3f m;

    m.load(r1, r2, r3);

    return m;
}

typedef quaternion quat;


/*template<typename T>
vec<3, T> operator*(const mat<3, T> m, const vec<3, T>& other)
{
    vec<3, T> val;

    val.v[0] = m.v[0][0] * other.v[0] + m.v[0][1] * other.v[1] + m.v[0][2] * other.v[2];
    val.v[1] = m.v[1][0] * other.v[0] + m.v[1][1] * other.v[1] + m.v[1][2] * other.v[2];
    val.v[2] = m.v[2][0] * other.v[0] + m.v[2][1] * other.v[1] + m.v[2][2] * other.v[2];

    return val;
}*/


#endif // VEC_HPP_INCLUDED
