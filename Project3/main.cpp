#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"



Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
	Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f proj, ortho;
    eye_fov = (eye_fov / 180.0) * MY_PI; // ת������

    proj << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;         //͸��ͶӰ����

    double w, h, z;
    h = -zNear * tan(eye_fov / 2) * 2;
    w = h * aspect_ratio;
    z = zFar - zNear;

    ortho << 2 / w, 0, 0, 0,
        0, 2 / h, 0, 0,
        0, 0, 2 / z, -(zFar + zNear) / 2,
        0, 0, 0, 1; //����ͶӰ������Ϊ�ڹ۲�ͶӰʱx0yƽ���ӽ�Ĭ�������ģ��������������ͶӰ�Ͳ���ƽ��x��y��

    projection = ortho * proj * projection;

	return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        //���� ֱ�����ô����payload��texture���Խ��е���getcolor()����
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
        //����Ҳ�����Զ����˫���Բ�ֵ��׼ȷ�Ļ��������ɫ
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);   //�����⡢������͸߹�ϵ��

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};     //�ƹ�ǿ�ȡ�λ��

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};    //������ǿ��
    Eigen::Vector3f eye_pos{0, 0, 10};      //�۾��۲�λ��

    float p = 150;

    Eigen::Vector3f color = texture_color;      //������ɫ
    Eigen::Vector3f point = payload.view_pos;   //��ɫ������
    Eigen::Vector3f normal = payload.normal;    //��ɫ�㷨����
    Eigen::Vector3f amb, dif, spe, l, v;        //amb :�����⣻  dif :�����䣻  spe: �߹�



    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        l = (light.position - point).normalized();   // lΪ���ߴ�ĳ��ķ������� ���ù�Դλ����ĳ�㹹������
        v = (eye_pos - point).normalized() + l;      //�۲ⷽ�� �۾��۲췽���λ�� ����ɫ�����깹������  ����1 ��һ������ֹ����

        //�����乫ʽ �� kd(I/r*r)max(0, n.dot(l))
        dif = kd.cwiseProduct(light.intensity / ((light.position - point).dot(light.position - point))) * std::fmax(0, normal.dot(l));  //norma ���Ƿ�����n
        // �߹⹫ʽ�� kd(I/r*r)max(0, n.dot(h))
        spe = ks.cwiseProduct(light.intensity / ((light.position - point).dot(light.position - point))) * pow(std::fmax(0, normal.dot(v.normalized())), p);
        // �����⹫ʽ�� ka*Ia
        amb = ka.cwiseProduct(amb_light_intensity);
        result_color += (dif + spe + amb);  //ÿ�������ɫ �����ֹ����

    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};      //�ƹ�ǿ�ȡ�λ��
    Eigen::Vector3f amb_light_intensity{10, 10, 10};        //������ǿ��
    Eigen::Vector3f eye_pos{0, 0, 10};   //�۾��۲�λ��

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    
    Eigen::Vector3f amb, dif, spe, l, v;

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        l = (light.position - point).normalized();   // lΪ���ߴ�ĳ��ķ������� ���ù�Դλ����ĳ�㹹������
        v = (eye_pos - point).normalized() + l;      //�۲ⷽ�� �۾��۲췽���λ�� ����ɫ�����깹������  ����1 ��һ������ֹ����

        //�����乫ʽ �� kd(I/r*r)max(0, n.dot(l))
        dif = kd.cwiseProduct(light.intensity / ((light.position - point).dot(light.position - point))) * std::fmax(0, normal.dot(l));  //norma ���Ƿ�����n
        // �߹⹫ʽ�� kd(I/r*r)max(0, n.dot(h))
        spe = ks.cwiseProduct(light.intensity / ((light.position - point).dot(light.position - point))) * pow(std::fmax(0, normal.dot(v.normalized())), p);
        // �����⹫ʽ�� ka*Ia
        amb = ka.cwiseProduct(amb_light_intensity);
        result_color += (dif + spe + amb);  //ÿ�������ɫ �����ֹ����
    }

    return result_color * 255.f;
}



Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)    // û����
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)
    float x, y, z;
    Vector3f t, b;
    x = normal.x(), y = normal.y(), z = normal.z();
    t << x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z* y / sqrt(x * x + z * z);
    b = normal.cross(t);

    Matrix3f TBN;
    TBN << t.x(), b.x(), normal.x(),
        t.y(), b.y(), normal.y(),
        t.z(), b.z(), normal.z();

    float u, v, w, h;
    u = payload.tex_coords.x();
    v = payload.tex_coords.y();
    w = payload.texture->width;
    h = payload.texture->height;

    float dU = kh * kn * (payload.texture->getColorBilinear(u + 1.0 / w, v).norm() - payload.texture->getColorBilinear(u, v).norm());
    float dV = kh * kn * (payload.texture->getColorBilinear(u, v + 1.0 / h).norm() - payload.texture->getColorBilinear(u, v).norm());

    Vector3f ln;
    ln << -dU, dV, 1;

    point += kn * normal * payload.texture->getColorBilinear(u, v).norm();//��һ����������displacement��bump�Ĺؼ���λ��
    normal = (TBN * ln).normalized();
    

    Eigen::Vector3f result_color = {0, 0, 0};
    Eigen::Vector3f amb, dif, spe, l, v1;
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        l = (light.position - point).normalized();
        v1 = (eye_pos - point).normalized() + l;
        dif = kd.cwiseProduct(light.intensity / ((light.position - point).dot(light.position - point))) * std::fmax(0, normal.dot(l));
        spe = ks.cwiseProduct(light.intensity / ((light.position - point).dot(light.position - point))) * pow(std::fmax(0, normal.dot(v1.normalized())), p);
        amb = ka.cwiseProduct(amb_light_intensity);
        result_color += (dif + spe + amb);

    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)    //��û��ô����
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)

    float x, y, z;
    Vector3f t, b;
    x = normal.x(), y = normal.y(), z = normal.z();
    t << x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z* y / sqrt(x * x + z * z);
    b = normal.cross(t);

    Matrix3f TBN;
    TBN << t.x(), b.x(), normal.x(),
        t.y(), b.y(), normal.y(),
        t.z(), b.z(), normal.z();

    float u, v, w, h;
    u = payload.tex_coords.x();
    v = payload.tex_coords.y();
    w = payload.texture->width;
    h = payload.texture->height;

    float dU = kh * kn * (payload.texture->getColorBilinear(u + 1.0 / w, v).norm() - payload.texture->getColorBilinear(u, v).norm());
    float dV = kh * kn * (payload.texture->getColorBilinear(u, v + 1.0 / h).norm() - payload.texture->getColorBilinear(u, v).norm());

    Vector3f ln;
    ln << -dU, dV, 1;

    normal = (TBN * ln).normalized();



    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "./models/spot/";

    // Load .obj File
    //  ��ȡָ��·����ģ���ļ������ϳ������εļ�����Ϊ��Ⱦ���ߵ����루��������������Ķ���Σ�
    bool loadout = Loader.LoadFile("./models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";    // ��ѡ�� texture������ͼʱ���޸�ͼƬΪ spot_texture.png
    r.set_texture(Texture(obj_path + texture_path));

    //������һ��fragment_shader_payload�����������л�ȡҪʹ�õ���ɫ����
    //���payload������ Fragment Shader �����õ��Ĳ������Լ�Ҫʹ�õ�shader���ͣ�
    //����ͨ���Ա�֤�˵��ò�ͬ��shaderʱ�������ظ�д��ͬ�Ĵ���
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = bump_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;                //��ɫģ��
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {

        //draw�����������ݽ���դ������ʽ��ʼ��Ⱦ��������
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));


        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);

        //��դ�������е������󣬻ص�main����������opencv�Դ��ķ�����frame_buffer��ȡ������õ����ݻ���ͼ�в��洢��
        //�Դˣ���Ⱦ���ߵ����̽���
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);
        std::cout << "frame count: " << frame_count++ << '\n';		//��ʾ��ǰΪ�ڼ�֡
       
        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}