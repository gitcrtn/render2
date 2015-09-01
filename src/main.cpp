// render2
// Keita Yamada
// 2015.08.29

#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <array>
#include <iostream>
#include <math.h>
#include <mutex>
#include <thread>

#include "argparse.h"

std::mutex mtx_image;

class Vec3d{
public:
	double x, y, z;

	inline Vec3d(double x, double y, double z) :x(x), y(y), z(z) {}
	inline Vec3d() :x(0), y(0), z(0) {}

	inline bool operator == (const Vec3d& v) const
	{
		if (this->x == v.x && this->y == v.y && this->z == v.z) return true;
		else return false;
	}

	inline bool operator != (const Vec3d& v) const
	{
		return (*(this) == v);
	}

	inline double length() const
	{
		return sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
	}

	inline double dot(const Vec3d& v) const
	{
		return this->x * v.x + this->y * v.y + this->z * v.z;
	}

	inline double operator ^ (const Vec3d& v) const
	{
		return this->dot(v);
	}

	inline Vec3d cross(const Vec3d &v) const
	{
		return Vec3d(this->y * v.z - this->z * v.y, this->z * v.x - this->x * v.z, this->x * v.y - this->y * v.x);
	}

	inline Vec3d operator % (const Vec3d& v) const
	{
		return this->cross(v);
	}
	
	inline const Vec3d& operator %= (const Vec3d& v)
	{
		this->x = this->y * v.z - this->z * v.y;
		this->y = this->z * v.x - this->x * v.z;
		this->z = this->x * v.y - this->y * v.x;
		return *(this);
	}

	inline const Vec3d& operator += (const Vec3d& v)
	{
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
		return *(this);
	}

	inline Vec3d operator + (const Vec3d& v) const
	{
		return Vec3d(this->x + v.x, this->y + v.y, this->z + v.z);
	}

	inline const Vec3d& operator -= (const Vec3d& v)
	{
		this->x -= v.x;
		this->y -= v.y;
		this->z -= v.z;
		return *(this);
	}

	inline operator bool() const
	{
		return this->x && this->y && this->z;
	}

	inline Vec3d operator - (const Vec3d& v) const
	{
		return Vec3d(this->x - v.x, this->y - v.y, this->z - v.z);
	}

	inline const Vec3d& operator - ()
	{
		this->x *= -1.0;
		this->y *= -1.0;
		this->z *= -1.0;
		return *(this);
	}

	inline const Vec3d& operator *= (const Vec3d& v)
	{
		this->x *= v.x;
		this->y *= v.y;
		this->z *= v.z;
		return *(this);
	}

	inline const Vec3d& operator *= (double a)
	{
		this->x *= a;
		this->y *= a;
		this->z *= a;
		return *(this);
	}

	inline Vec3d operator * (const Vec3d& v) const
	{
		return Vec3d(this->x * v.x, this->y * v.y, this->z * v.z);
	}

	inline Vec3d operator * (double a) const
	{
		return Vec3d(this->x * a, this->y * a, this->z * a);
	}

	inline Vec3d operator * (float a) const
	{
		return Vec3d(this->x * a, this->y * a, this->z * a);
	}

	inline const Vec3d& operator /= (const Vec3d& v)
	{
		this->x /= v.x;
		this->y /= v.y;
		this->z /= v.z;
		return *(this);
	}

	inline const Vec3d& operator /= (double a)
	{
		this->x /= a;
		this->y /= a;
		this->z /= a;
		return *(this);
	}

	inline Vec3d operator / (const Vec3d& v) const
	{
		return Vec3d(this->x / v.x, this->y / v.y, this->z / v.z);
	}

	inline Vec3d operator / (double a) const
	{
		return Vec3d(this->x / a, this->y / a, this->z / a);
	}

	inline Vec3d operator / (int a) const
	{
		return Vec3d(this->x / a, this->y / a, this->z / a);
	}

	inline const Vec3d& normalize()
	{
		double len = this->length();
		this->x /= len;
		this->y /= len;
		this->z /= len;
		return *(this);
	}

	inline Vec3d normalized() const
	{
		double len = this->length();
		return Vec3d(this->x / len, this->y / len, this->z / len);
	}

};

/*
Vec3d double::operator * (Vec3d& v)
{
	
}
*/

typedef std::vector<Vec3d> ImageBuffer;

// camera
class Camera{
public:
	Vec3d pos, dir, up;

	Camera(Vec3d pos, Vec3d dir, Vec3d up)
	{
		this->pos = pos;
		this->dir = dir;
		this->up = up;
	}
};

// render settings
class Setting{
public:
	int reso_w, reso_h, samples, supersamples, ipr_x1, ipr_y1, ipr_x2, ipr_y2, depth, iteration;
	bool time_over;

	Setting(int w, int h, int samples, int supersamples, int depth, int ipr_x, int ipr_y, int ipr_w, int ipr_h)
	{
		this->reso_w = w;
		this->reso_h = h;
		this->samples = samples;
		this->supersamples = supersamples;
		this->depth = depth;
		this->ipr_x1 = ipr_x;
		this->ipr_y1 = ipr_y;
		this->ipr_x2 = ipr_x + ipr_w;
		this->ipr_y2 = ipr_y + ipr_h;
		this->iteration = 0;
		this->time_over = false;
	}
};

// screen
class Screen{
public:
	double width,height,dist;
	Vec3d x, y, center;

	Screen(const Camera& camera, const Setting& setting, double scale=30.0, double dist=40.0)
	{
		this->width = scale * setting.reso_w / setting.reso_h;
		this->height = scale;
		this->dist = dist;
		this->x = camera.dir.cross(camera.up).normalized() * this->width;
		this->y = camera.dir.cross(this->x).normalized()   * this->height;
		this->center = camera.pos + camera.dir * this->dist;
	}
};

// ray
class Ray{
public:
	Vec3d pos, dir;

	Ray(const Camera& camera, const Screen& screen, const Setting& setting, int x, int y, int supersample_x, int supersample_y)
	{
		double rate = 1.0 / setting.supersamples;
		double rx = supersample_x * rate + rate / 2.0;
		double ry = supersample_y * rate + rate / 2.0;
		Vec3d end = screen.center + screen.x * ((rx + x) / setting.reso_w - 0.5) + screen.y * ((ry + y) / setting.reso_h - 0.5);
		this->pos = camera.pos;
		this->dir = (end - this->pos).normalized();
	}

	Ray(Vec3d pos, Vec3d dir)
	{
		this->pos = pos;
		this->dir = dir;
	}

	Vec3d line(double distance)
	{
		return this->pos + this->dir * distance;
	}
};

// selection of reflection algorithm
enum Shader
{
	Diffuse, Reflection, Refraction
};

class Material{
public:
	Vec3d emission;
	Vec3d diffuse;
	Vec3d specular;  // not yet used
	Vec3d reflection;
	Vec3d refraction;
	Shader shader;
	Material() :diffuse(Vec3d()),specular(Vec3d()),reflection(Vec3d()),refraction(Vec3d()),emission(Vec3d()),shader(Shader::Diffuse){}
};

class Intersection;

// (interface) instance of mesh
class MeshObj{
public:
	Vec3d pos;
	Material* mat;

	MeshObj(Vec3d pos, Material* mat = nullptr)
	{
		this->pos = pos;
		this->mat = mat;
	}

	virtual void intersect(Intersection& intersection)
	{
		return;
	}

	virtual Vec3d normal(Vec3d& position)
	{
		return Vec3d();
	}
};

class Intersection{
public:
	Ray& ray;
	MeshObj* obj;
	Vec3d point, normal;

	Intersection(Ray& ray) :ray(ray), obj(nullptr),_distance(INFINITY),point(Vec3d()),normal(Vec3d()){}

	void update(double distance, MeshObj* obj)
	{
		if (distance < this->_distance)
		{
			this->_distance = distance;
			this->obj = obj;
		}
	}

	bool final()
	{
		if (this->_distance != INFINITY)
		{
			this->_point();
			this->_normal();
			return true;
		}
		else
			return false;
	}

private:

	double _distance;

	void _point()
	{
		this->point = this->ray.line(this->_distance);
	}

	void _normal()
	{
		this->normal = this->obj->normal(this->point);
	}	
};

// sphere
class Sphere : public MeshObj{

public:
	double radius;
		
	Sphere(Vec3d pos, double radius) :MeshObj(pos), radius(radius){}

	virtual void intersect(Intersection& intersection)
	{
		bool doesIntersect = true;

		Vec3d v = intersection.ray.pos - this->pos;
		double B = 2.0 * (intersection.ray.dir ^ v);
		double C = (v ^ v) - (radius * radius);

		double t = 0.0;

		// compute discriminant
		// if negative, there is no intersection

		double discr = B*B - 4.0 * C;

		if (discr < 0.0)
		{
			// line and Sphere3 do not intersect

			doesIntersect = false;
		}
		else
		{
			// t0: (-B - sqrt(B^2 - 4AC)) / 2A  (A = 1)

			double sqroot = sqrt(discr);
			t = (-B - sqroot) * 0.5;

			if (t < 0.0)
			{
				// no intersection, try t1: (-B + sqrt(B^2 - 4AC)) / 2A  (A = 1)

				t = (-B + sqroot) * 0.5;
			}

			if (t < 0.0)
				doesIntersect = false;
		}

		if (doesIntersect) intersection.update(t,this);
	}

	virtual Vec3d normal(Vec3d& position)
	{
		return Vec3d(position - this->pos).normalized();
	}
};

class Cylinder : public MeshObj{

public:
	double radius;
	//Vec3d end;  
	double height;

	Cylinder(Vec3d pos, double height, double radius) :MeshObj(pos), height(height), radius(radius){}

	virtual void intersect(Intersection& intersection)
	{
		bool doesIntersect;
		Vec3d upVector = Vec3d(0, 1, 0);
		double target = INFINITY;
		Vec3d alpha = upVector * intersection.ray.dir.dot(upVector);
		Vec3d deltaP = intersection.ray.pos - this->pos;
		Vec3d beta = upVector * deltaP.dot(upVector);
		Vec3d center2 = this->pos + upVector * this->height;

		double a = (intersection.ray.dir - alpha).length();
		double b = 2.0 * ((intersection.ray.dir - alpha).dot(deltaP-beta));
		double c = (deltaP - beta).length() - this->radius * this->radius;

		double discr = b*b - 4 * a*c;
		if (discr < 0) doesIntersect = false;
		else
		{
			discr = sqrt(discr);
			double t1 = ((-1 * b) + discr) / (2 * a);
			double t2 = ((-1 * b) - discr) / (2 * a);
			if (t1 >= 0)
			{
				if (t1 < target && upVector.dot((intersection.ray.dir - this->pos) + intersection.ray.dir * t1) > 0 && upVector.dot((intersection.ray.pos - center2) + intersection.ray.dir*t1) < 0)
				{
					target = t1;
					doesIntersect = true;
				}
			}
			if (t2 >= 0)
			{
				if (t2 < target && upVector.dot((intersection.ray.dir - this->pos) + intersection.ray.dir * t2) > 0 && upVector.dot((intersection.ray.pos - center2) + intersection.ray.dir*t2) < 0)
				{
					target = t2;
					doesIntersect = true;
				}
			}

		}

		float denom = intersection.ray.dir.dot(upVector);
		if (denom > 1e-6)
		{
			Vec3d co = this->pos - intersection.ray.pos;
			double t3 = co.dot(upVector) / denom;
			if (t3 > 0 && t3 < target && (intersection.ray.dir * t3 - co).length() <= this->radius * this->radius)
			{
				target = t3;
				doesIntersect = true;
			}
			
		}
		else if (denom < 1e-6)
		{
			Vec3d co2 = center2 - intersection.ray.pos;
			double t4 = co2.dot(upVector) / denom;
			if (t4 > 0 && t4 < target && (intersection.ray.dir * t4 - co2).length() <= this->radius * this->radius)
			{
				target = t4;
				doesIntersect = true;
			}
		}
		if (doesIntersect) intersection.update(target, this);

	}

	virtual Vec3d normal(Vec3d& position)
	{
		Vec3d upVector(0, 1, 0);
		if (abs((position - this->pos).dot(upVector)) < 1e-4)
		{
			return -upVector.normalized();
		}
		else
		{
			Vec3d top = this->pos + upVector * height;
			Vec3d perp = (-upVector).cross(position - top);
			return ((position - top).cross(perp)).normalized();
		}
	}
};

class Circle : public MeshObj{

public:
	double radius;
	Vec3d dir;

	Circle(Vec3d pos, Vec3d dir, double radius) :MeshObj(pos), dir(dir), radius(radius){}

	virtual void intersect(Intersection& intersection)
	{

	}

	virtual Vec3d normal(Vec3d& position)
	{
		return Vec3d();
	}
};

class Square : public MeshObj{

public:
	double w,h;
	Vec3d dir;

	Square(Vec3d pos, Vec3d dir, double w, double h) :MeshObj(pos), dir(dir), w(w), h(h){}

	virtual void intersect(Intersection& intersection)
	{

	}

	virtual Vec3d normal(Vec3d& position)
	{
		return Vec3d();
	}
};


// scene
class Scene{
public:

	//std::vector<Geometry> geometries;
	//Imath::C3f bgColor;
	std::vector<MeshObj*> objects;
	Vec3d bgColor;

	Scene(Vec3d bgColor = Vec3d())
	{
		//geometries.clear();
		this->objects.clear();
		this->bgColor = bgColor;
	}

	bool test_intersect(Intersection& intersection)
	{
		for (auto &o : this->objects)
		{
			o->intersect(intersection);
		}
		return intersection.final();
	}
};

// Xor-Shiftによる乱数ジェネレータ
class XorShift {
	unsigned int seed_[4];
public:
	unsigned int next(void) {
		const unsigned int t = seed_[0] ^ (seed_[0] << 11);
		seed_[0] = seed_[1];
		seed_[1] = seed_[2];
		seed_[2] = seed_[3];
		return seed_[3] = (seed_[3] ^ (seed_[3] >> 19)) ^ (t ^ (t >> 8));
	}

	double next01(void) {
		return (double)next() / UINT_MAX;
	}

	XorShift(const unsigned int initial_seed) {
		unsigned int s = initial_seed;
		for (int i = 1; i <= 4; i++){
			seed_[i - 1] = s = 1812433253U * (s ^ (s >> 30)) + i;
		}
	}
};

typedef XorShift Random;

Vec3d radiance(Scene& scene, Ray& ray, Random& rnd, const int depth, const int max_depth)
{
	Intersection intersection(ray);

	if (!scene.test_intersect(intersection))
	{
		return scene.bgColor;
	}

	if (depth > max_depth)
	{
		return intersection.obj->mat->emission;
	}

	switch (intersection.obj->mat->shader)
	{
	case Shader::Diffuse: if (!intersection.obj->mat->diffuse) return intersection.obj->mat->emission; else break;
	case Shader::Reflection: if (!intersection.obj->mat->reflection) return intersection.obj->mat->emission; else break;
	case Shader::Refraction: if (!intersection.obj->mat->refraction && !intersection.obj->mat->reflection) return intersection.obj->mat->emission; else break;
	default:break;
	}

	Vec3d orienting_normal = intersection.normal;

	if ((intersection.normal ^ ray.dir) > 0.0)
	{
		orienting_normal *= -1.0;
	}

	float reflect_ratio = (5.0f - static_cast<float>(depth)) / 5.0f;

	Vec3d rad(0,0,0), weight(1,1,1), inc_rad(1,1,1);
	bool single_radiance = true;

	switch (intersection.obj->mat->shader)
	{
	case Shader::Diffuse:
	{
		Vec3d w, u, v;
		w = orienting_normal;
		if (fabs(w.x) > 0.0000009)
		{
			u = (Vec3d(0,1,0) % w).normalized();
		}
		else
		{
			u = (Vec3d(1,0,0) % w).normalized();
		}
		v = w % u;
		double r1 = 2.0 * M_PI * rnd.next01();
		double r2 = rnd.next01();
		double rr2 = sqrt(r2);
		Ray new_ray(intersection.point, (u * cos(r1) * rr2 + v * sin(r1) * rr2 + w * sqrt(1.0 - r2)).normalized());
		rad = radiance(scene, new_ray, rnd, depth + 1, max_depth);
		weight = intersection.obj->mat->diffuse * reflect_ratio;
	}
		break;

	case Shader::Reflection:
	{
		Ray new_ray(intersection.point, ray.dir - intersection.normal * 2.0 * (intersection.normal ^ ray.dir));
		rad = radiance(scene, new_ray, rnd, depth + 1, max_depth);
		weight = intersection.obj->mat->reflection * reflect_ratio;
	}
		break;

	case Shader::Refraction:
		bool into = (orienting_normal ^ intersection.normal) > 0.0;

		double default_refraction = 1.0;
		double object_refraction = 1.5;
		double ray_refraction;
		if (into)
		{
			ray_refraction = default_refraction / object_refraction;
		}
		else
		{
			ray_refraction = object_refraction / default_refraction;
		}
		double incident_dot = ray.dir ^ orienting_normal;
		double critical_factor = 1.0 - pow(ray_refraction, 2) * (1.0 - pow(incident_dot,2));

		Ray reflection_ray(intersection.point, ray.dir - intersection.normal * 2.0 * (intersection.normal ^ ray.dir));

		// total reflection
		if (critical_factor < 0.0)
		{
			if (!intersection.obj->mat->reflection) return intersection.obj->mat->emission;
			rad = radiance(scene, reflection_ray, rnd, depth + 1, max_depth);
			weight = intersection.obj->mat->reflection * reflect_ratio;
			break;
		}

		Ray refraction_ray(intersection.point, (ray.dir * ray_refraction - intersection.normal * (into ? 1.0 : -1.0) * (incident_dot * ray_refraction + sqrt(critical_factor))).normalized());

		double a = object_refraction - default_refraction;
		double b = object_refraction + default_refraction;
		double vertical_incidence_factor = pow(a, 2) / pow(b, 2);
		double c = 1.0 - (into ? -1.0 * incident_dot : (refraction_ray.dir ^ -orienting_normal));
		double fresnel_incidence_factor = vertical_incidence_factor + (1.0 - vertical_incidence_factor) * pow(c,5);
		double radiance_scale = pow(ray_refraction, 2.0);
		double refraction_factor = (1.0 - fresnel_incidence_factor) * radiance_scale;

		double probability = 0.75 + fresnel_incidence_factor;
		if (depth > 2)
		{
			if (rnd.next01() < probability)
			{
				if (!intersection.obj->mat->reflection) return intersection.obj->mat->emission;
				rad = radiance(scene, reflection_ray, rnd, depth + 1, max_depth) * fresnel_incidence_factor;
				weight = intersection.obj->mat->reflection * reflect_ratio;
			}
			else
			{
				if (!intersection.obj->mat->refraction) return intersection.obj->mat->emission;
				rad = radiance(scene, refraction_ray, rnd, depth + 1, max_depth) * refraction_factor;
				weight = intersection.obj->mat->refraction * reflect_ratio;
			}
		}
		else
		{
			single_radiance = false;

			// reflection radiance
			if (!intersection.obj->mat->reflection && intersection.obj->mat->refraction) inc_rad = intersection.obj->mat->emission;
			else
			{
				rad = radiance(scene, reflection_ray, rnd, depth + 1, max_depth) * fresnel_incidence_factor;
				weight = intersection.obj->mat->reflection * reflect_ratio;
				inc_rad = rad * weight;
			}

			// refraction radiance
			if (intersection.obj->mat->reflection && !intersection.obj->mat->refraction) inc_rad += intersection.obj->mat->emission;
			else
			{
				rad = radiance(scene, refraction_ray, rnd, depth + 1, max_depth) * refraction_factor;
				weight = intersection.obj->mat->refraction * reflect_ratio;
				inc_rad += rad * weight;
			}
		}		

		break;
	}

	if (single_radiance) inc_rad = rad * weight;

	return intersection.obj->mat->emission + inc_rad;

}

typedef struct tagBITMAPFILEHEADER {
	unsigned short bfType;
	unsigned long  bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	unsigned long  bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER{
	unsigned long  biSize;
	long           biWidth;
	long           biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned long  biCompression;
	unsigned long  biSizeImage;
	long           biXPixPerMeter;
	long           biYPixPerMeter;
	unsigned long  biClrUsed;
	unsigned long  biClrImporant;
} BITMAPINFOHEADER;

inline double clamp(double value)
{
	if (value < 0.0) return 0.0;
	else if (1.0 < value) return 1.0;
	else return value;
}

unsigned char d2c(double value)
{
	return static_cast<unsigned char>(clamp(value) * 255);
}

void write_bmp(const char* imagename, ImageBuffer& buffer, const Setting& setting)
{
	unsigned int scan_line_bytes = setting.reso_w * 3;
	int file_size = sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER)+scan_line_bytes * setting.reso_h;
	BITMAPFILEHEADER header;
	header.bfType = 'B' | ('M' << 8);
	header.bfSize = file_size;
	header.bfReserved1 = 0;
	header.bfReserved2 = 0;
	header.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	BITMAPINFOHEADER infoHeader;
	infoHeader.biSize = sizeof(BITMAPINFOHEADER);
	infoHeader.biWidth = setting.reso_w;
	infoHeader.biHeight = setting.reso_h;
	infoHeader.biPlanes = 1;
	infoHeader.biBitCount = 24;
	infoHeader.biCompression = 0;
	infoHeader.biSizeImage = setting.reso_w * setting.reso_h * 3;
	infoHeader.biXPixPerMeter = 3780;
	infoHeader.biYPixPerMeter = 3780;
	infoHeader.biClrUsed = 0;
	infoHeader.biClrImporant = 0;
	
	FILE *fp = fopen(imagename, "wb");
	int i, j, k, index;
	unsigned char buf;

	fwrite(&header.bfType, sizeof(header.bfType), 1, fp);
	fwrite(&header.bfSize, sizeof(header.bfSize), 1, fp);
	fwrite(&header.bfReserved1, sizeof(header.bfReserved1), 1, fp);
	fwrite(&header.bfReserved2, sizeof(header.bfReserved2), 1, fp);
	fwrite(&header.bfOffBits, sizeof(header.bfOffBits), 1, fp);

	fwrite(&infoHeader.biSize, sizeof(infoHeader.biSize), 1, fp);
	fwrite(&infoHeader.biWidth, sizeof(infoHeader.biWidth), 1, fp);
	fwrite(&infoHeader.biHeight, sizeof(infoHeader.biHeight), 1, fp);
	fwrite(&infoHeader.biPlanes, sizeof(infoHeader.biPlanes), 1, fp);
	fwrite(&infoHeader.biBitCount, sizeof(infoHeader.biBitCount), 1, fp);
	fwrite(&infoHeader.biCompression, sizeof(infoHeader.biCompression), 1, fp);
	fwrite(&infoHeader.biSizeImage, sizeof(infoHeader.biSizeImage), 1, fp);
	fwrite(&infoHeader.biXPixPerMeter, sizeof(infoHeader.biXPixPerMeter), 1, fp);
	fwrite(&infoHeader.biYPixPerMeter, sizeof(infoHeader.biYPixPerMeter), 1, fp);
	fwrite(&infoHeader.biClrUsed, sizeof(infoHeader.biClrUsed), 1, fp);
	fwrite(&infoHeader.biClrImporant, sizeof(infoHeader.biClrImporant), 1, fp);

	for (i = 0; i < (int)setting.reso_h; i++)
	{

		for (j = 0; j < (int)setting.reso_w; j++) {
			index = (setting.reso_h - i - 1) * setting.reso_w + j;
			buf = d2c(buffer[index].y / setting.iteration);
			fwrite(&buf, 1, 1, fp);
			buf = d2c(buffer[index].x / setting.iteration);
			fwrite(&buf, 1, 1, fp);
			buf = d2c(buffer[index].z / setting.iteration);
			fwrite(&buf, 1, 1, fp);
		}
	}
	fclose(fp);	
}



void write_ppm(const char* imagename, ImageBuffer& buffer, const Setting& setting)
{
	FILE *f;
	f = fopen(imagename, "wb");
	fprintf(f, "P3\n%d %d\n%d\n", setting.reso_w, setting.reso_h, 255);
	for (int i = 0; i < setting.reso_w * setting.reso_h; i++){
		fprintf(f, "%d %d %d ", d2c(buffer[i].x / setting.iteration), d2c(buffer[i].y / setting.iteration), d2c(buffer[i].z / setting.iteration));
	}

	fclose(f);
}

void dump_image(Setting& setting, ImageBuffer& buffer)
{
	char fname[255];
	int time_count = 1, time_max = 30;
	
	while (true)
	{
		std::this_thread::sleep_for(std::chrono::seconds(30));

		mtx_image.lock();

		std::cout << "\n" << time_count * 0.5f << "minute(s)" << std::endl;
		std::cout << "image output..." << std::endl;
		sprintf(fname, "out_%02d.bmp", time_count);
		write_bmp(fname, buffer, setting);
		//write_ppm(fname, buffer, setting);

		mtx_image.unlock();

		time_count++;
		if (time_count > time_max)
		{
			setting.time_over = true;
			break;
		}
	}
}

void render(Setting& setting, Camera& camera, Screen& screen, Scene& scene, ImageBuffer& buffer)
{
	setting.iteration = 1;	

	std::thread t(dump_image, std::ref(setting), std::ref(buffer));

	//std::vector<Vec3d> local_buf(setting.reso_w * setting.reso_h * setting.supersamples * setting.supersamples);
	std::vector<Vec3d> local_buf(setting.reso_w * setting.reso_h);
	for (Vec3d& v : local_buf) v = Vec3d();

	while (!setting.time_over)
	{

#pragma omp parallel for schedule(dynamic, 1)       // OpenMP 
		for (int y = 0; y < setting.reso_h; y++)
		{
			Random rnd(setting.iteration + y + 1);
			printf("\rRendering (y = %d) %3.2f%% [iteration: %d]    ", y, (100.0 * y / (setting.reso_h - 1)),setting.iteration);

			for (int x = 0; x < setting.reso_w; x++)
			{
				int index = (setting.reso_h - y - 1) * setting.reso_w + x;

				if (!(setting.ipr_x1 <= x && x <= setting.ipr_x2 && setting.ipr_y1 <= y && y <= setting.ipr_y2)) continue;

				for (int sy = 0; sy < setting.supersamples; sy++)
				{
					for (int sx = 0; sx < setting.supersamples; sx++)
					{
						Vec3d acm_rad;

						//int index = (setting.reso_h - y - 1) * setting.reso_w + x + 2 * sy + sx;

						for (int s = 0; s < setting.samples; s++)
						{
							Ray ray = Ray(camera, screen, setting, x, y, sx, sy);
							acm_rad += radiance(scene, ray, rnd, 0, setting.depth) / setting.samples / (setting.supersamples * setting.supersamples);
							local_buf[index] += acm_rad;
						}
					}
				}
			}	
		}

		if (setting.time_over) break;

		setting.iteration++;

		std::lock_guard<std::mutex> lock(mtx_image); 
		for (int y=0; y < setting.reso_h; y++)
		{
			for (int x=0; x < setting.reso_w; x++)
			{
				int index = (setting.reso_h - y - 1) * setting.reso_w + x;
				//buffer[index] = local_buf[index] + local_buf[index+1] + local_buf[index+2] + local_buf[index+3];
				buffer[index] = local_buf[index];
			}
		}
	}
	t.join();
}

int main(int argc, char** argv)
{
	ArgParser parser(argc, argv);
	parser.request(ARG_TYPE::INT, "-x", "width", "resolution width", 1024);
	parser.request(ARG_TYPE::INT, "-y", "height", "resolution height", 768);
	parser.request(ARG_TYPE::INT, "-s", "sample", "ray samples per pixel", 16);
	parser.request(ARG_TYPE::INT, "-p", "subpixel", "pixel resolution for anti-aliasing", 4);
	parser.request(ARG_TYPE::INT, "-d", "depth", "ray trace depth", 5);

	parser.request(ARG_TYPE::INT, "-ipr_x", "ipr_x", "IPR start x", 0);
	parser.request(ARG_TYPE::INT, "-ipr_y", "ipr_y", "IPR start y", 0);
	parser.request(ARG_TYPE::INT, "-ipr_w", "ipr_w", "IPR Rect w", 1024);
	parser.request(ARG_TYPE::INT, "-ipr_h", "ipr_h", "IPR Rect h", 768);

	parser.parse();
	parser.print_all();

	Setting render_settings(
		parser.get("width"), parser.get("height"), parser.get("sample"), parser.get("subpixel"), parser.get("depth"),
		parser.get("ipr_x"), parser.get("ipr_y"), parser.get("ipr_w"), parser.get("ipr_h"));
	Camera cam(Vec3d(50.0, 52.0, 220.0), Vec3d(0.0, -0.04, -30.0).normalized(), Vec3d(0.0, -1.0, 0.0));
	Screen screen(cam, render_settings);
	ImageBuffer image(render_settings.reso_w * render_settings.reso_h);
	for (Vec3d& v : image) v = Vec3d();

	Scene scene;

	Material black;
	Material red;		red.diffuse = Vec3d(0.75, 0.25, 0.25);
	Material blue;		blue.diffuse = Vec3d(0.25, 0.25, 0.75);
	Material green;		green.diffuse = Vec3d(0.25, 0.75, 0.25);
	Material gray;		gray.diffuse = Vec3d(0.75, 0.75, 0.75);
	Material glass;		glass.refraction = Vec3d(0.99, 0.99, 0.99);	glass.shader = Shader::Refraction;
	Material chrome;	chrome.reflection = Vec3d(0.99, 0.99, 0.99);	chrome.shader = Shader::Reflection;
	Material light;		light.emission = Vec3d(36, 36, 36);

	Sphere* a = new Sphere(Vec3d(1e5 + 1, 40.8, 81.6), 1e5); a->mat = &red; scene.objects.push_back(a);
	Sphere* b = new Sphere(Vec3d(-1e5 + 99, 40.8, 81.6), 1e5); b->mat = &blue; scene.objects.push_back(b);
	Sphere* c = new Sphere(Vec3d(50, 40.8, 1e5), 1e5); c->mat = &gray; scene.objects.push_back(c);
	Sphere* d = new Sphere(Vec3d(50, 40.8, -1e5 + 250), 1e5); d->mat = &black; scene.objects.push_back(d);
	Sphere* e = new Sphere(Vec3d(50, 1e5, 81.6), 1e5); e->mat = &gray; scene.objects.push_back(e);
	Sphere* f = new Sphere(Vec3d(50, -1e5 + 81.6, 81.6), 1e5); f->mat = &gray; scene.objects.push_back(f);
	Sphere* g = new Sphere(Vec3d(65, 20, 20), 20); g->mat = &green; scene.objects.push_back(g);
	Sphere* h = new Sphere(Vec3d(27, 16.5, 47), 16.5); h->mat = &chrome; scene.objects.push_back(h);
	Sphere* m = new Sphere(Vec3d(77, 16.5, 78), 16.5); m->mat = &glass; scene.objects.push_back(m);
	Sphere* n = new Sphere(Vec3d(50, 90, 81.6), 15); n->mat = &light; scene.objects.push_back(n);
	//Cylinder *o = new Cylinder(Vec3d(50, 16.5, 50), 20, 5); o->mat = &blue; scene.objects.push_back(o);

	render(render_settings,cam,screen,scene,image);
		
	return 0;
}