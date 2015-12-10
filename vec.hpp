#ifndef VEC_HPP_INCLUDED
#define VEC_HPP_INCLUDED

#include <math.h>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <float.h>

#define M_PI		3.14159265358979323846
#define M_PIf ((float)M_PI)

///bad, only for temporary debugging
#define EXPAND_3(vec) vec.v[0], vec.v[1], vec.v[2]

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

    vec() = default;

    vec(float val)
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

    vec<N, T> operator*(float other) const
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

    vec<N, T> operator/(float other) const
    {
        vec<N, T> r;

        for(int i=0; i<N; i++)
        {
            r.v[i] = v[i] / other;
        }

        return r;
    }

    float squared_length() const
    {
        float sqsum = 0;

        for(int i=0; i<N; i++)
        {
            sqsum += v[i]*v[i];
        }

        return sqsum;
    }


    float length() const
    {
        float l = squared_length();

        float val = sqrtf(l);

        return val;
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
        float len = length();

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
        vec<3, float> new_pos = this->rot(position, (vec<3, float>){-rotation.v[0], 0, 0});
        new_pos = new_pos.rot(position, (vec<3, float>){0, -rotation.v[1], 0});
        new_pos = new_pos.rot(position, (vec<3, float>){0, 0, -rotation.v[2]});

        return new_pos;
    }

    ///only valid for a 2-vec
    ///need to rejiggle the templates to work this out
    vec<2, T> rot(float rot_angle)
    {
        float len = length();

        float cur_angle = angle();

        float new_angle = cur_angle + rot_angle;

        float nx = len * cos(new_angle);
        float ny = len * sin(new_angle);

        return {nx, ny};
    }

    float angle()
    {
        return atan2(v[1], v[0]);
    }

    vec<3, T> get_euler() const
    {
        static_assert(N == 3, "Can only convert 3 element vectors into euler angles");

        vec<3, T> dir = *this;

        float cangle = dot((vec<3, T>){0, 1, 0}, dir.norm());

        float angle2 = acos(cangle);

        float y = atan2(dir.v[2], dir.v[0]);

        vec<3, T> rot = {0, y, angle2};

        return rot;
    }

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
    }
};

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
vec<N, T> randf()
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret.v[i] = randf_s();
    }

    return ret;
}

template<int N, typename T>
inline
vec<N, T> clamp(vec<N, T> v1, T p1, T p2)
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

template<int N, typename T>
inline float dot(const vec<N, T>& v1, const vec<N, T>& v2)
{
    float ret = 0;

    for(int i=0; i<N; i++)
    {
        ret += v1.v[i] * v2.v[i];
    }

    return ret;
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

template<int N, typename T>
inline vec<N, T> operator*(float v, const vec<N, T>& v1)
{
    return v1 * v;
}

template<int N, typename T>
inline vec<N, T> operator+(float v, const vec<N, T>& v1)
{
    return v1 + v;
}

/*inline vec3f operator-(float v, const vec3f& v1)
{
    return v1 - v;
}*/

///should convert these functions to be N/T

template<int N, typename T>
inline vec<N, T> operator/(float v, const vec<N, T>& v1)
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

template<typename U>
inline vec<3, float> xyz_to_vec(const U& xyz)
{
    vec<3, float> ret;

    ret.v[0] = xyz.x;
    ret.v[1] = xyz.y;
    ret.v[2] = xyz.z;

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

template<int N, typename T>
inline vec<N, T> d2r(const vec<N, T>& v1)
{
    vec<N, T> ret;

    for(int i=0; i<N; i++)
    {
        ret = (v1.v[i] / 360.f) * M_PI * 2;
    }

    return ret;
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

inline vec3f generate_flat_normal(const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
    return cross(p2 - p1, p3 - p1).norm();
}

///t should be some container of vec3f
///sorted via 0 -> about vector, plane perpendicular to that
template<typename T>
inline std::vector<vec3f> sort_anticlockwise(const T& in, const vec3f& about)
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


template<int N, typename T>
struct mat
{
    T v[N][N];

    mat<N, T> from_vec(vec3f v1, vec3f v2, vec3f v3) const
    {
        mat m;

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

    mat<N, T> invert() const
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

    void from_dir(vec3f dir)
    {
        vec3f up = {0, 1, 0};

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
        v[2][2] = dir.v[2];
    }

    void load_rotation_matrix(vec3f rotation)
    {
        vec3f c;
        vec3f s;

        for(int i=0; i<3; i++)
        {
            c.v[i] = cos(-rotation.v[i]);
            s.v[i] = sin(-rotation.v[i]);
        }

        ///rotation matrix
        vec3f r1 = {c.v[1]*c.v[2], -c.v[1]*s.v[2], s.v[1]};
        vec3f r2 = {c.v[0]*s.v[2] + c.v[2]*s.v[0]*s.v[1], c.v[0]*c.v[2] - s.v[0]*s.v[1]*s.v[2], -c.v[1]*s.v[0]};
        vec3f r3 = {s.v[0]*s.v[2] - c.v[0]*c.v[2]*s.v[1], c.v[2]*s.v[0] + c.v[0]*s.v[1]*s.v[2], c.v[1]*c.v[0]};

        load(r1, r2, r3);
    }

    vec<3, T> operator*(const vec<3, T>& other) const
    {
        vec<3, T> val;

        val.v[0] = v[0][0] * other.v[0] + v[0][1] * other.v[1] + v[0][2] * other.v[2];
        val.v[1] = v[1][0] * other.v[0] + v[1][1] * other.v[1] + v[1][2] * other.v[2];
        val.v[2] = v[2][0] * other.v[0] + v[2][1] * other.v[1] + v[2][2] * other.v[2];

        return val;
    }
};

/*template<typename T>
vec<3, T> operator*(const mat<3, T> m, const vec<3, T>& other)
{
    vec<3, T> val;

    val.v[0] = m.v[0][0] * other.v[0] + m.v[0][1] * other.v[1] + m.v[0][2] * other.v[2];
    val.v[1] = m.v[1][0] * other.v[0] + m.v[1][1] * other.v[1] + m.v[1][2] * other.v[2];
    val.v[2] = m.v[2][0] * other.v[0] + m.v[2][1] * other.v[1] + m.v[2][2] * other.v[2];

    return val;
}*/

typedef mat<3, float> mat3f;

#endif // VEC_HPP_INCLUDED
