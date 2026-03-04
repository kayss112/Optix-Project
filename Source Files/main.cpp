#include <optix.h>
#include "../Header Files/optix7.h"
#include "../gdt/gdt/gdt.h"
#include "../glfWindow/GLFWindow.h"
#include "../Header Files/SampleRenderer.h"
#include "../Header Files/LuanchParams.h"
#include "../Header Files/CUDABuffer.h"
#include <gl/GL.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

// embed_ptx 生成的 PTX 字符串
struct SampleWindow :public osc::GLFCameraWindow {
    SampleWindow(const std::string &title,const TriangleMesh &model
    ,const Camera &camera,const float worldScale) : osc::GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale),sample(model) {};
    virtual void render() override {
        if (cameraFrame.modified) {
            sample.setCamera(Camera{ cameraFrame.get_from(),cameraFrame.get_at(),cameraFrame.get_up() });

            cameraFrame.modified = false;
        }
        sample.render();
    }
    virtual void draw() override {
        //定义
        sample.downloadPixels(pixels.data());

        if (fbTexture == 0) { glGenTextures(1, &fbTexture); }
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        GLenum texFormat = GL_RGBA;
        GLenum texelType = GL_UNSIGNED_BYTE;
        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA, texelType, pixels.data());

        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 1);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glDisable(GL_DEPTH_TEST);

        glViewport(0, 0, fbSize.x, fbSize.y);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, (float)fbSize.x, 0.f, (fbSize).y, -1.f, 1.f);

        glBegin(GL_QUADS); 
        {
            glTexCoord2f(0.f, 0.f);
            glVertex3f(0.f, 0.f, 0.f);

            glTexCoord2f(0.f, 1.f);
            glVertex3f(0.f, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 1.f);
            glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

            glTexCoord2f(1.f, 0.f);
            glVertex3f((float)fbSize.x, 0.f, 0.f);
        }
        glEnd();
    }
    virtual void resize(const gdt::vec2i& newSize) {
        fbSize = newSize;
        sample.resize(newSize);
        pixels.resize(newSize.x * newSize.y);
    }
    gdt::vec2i fbSize;
    GLuint fbTexture{ 0 };
    SampleRenderer sample;
    std::vector<uint32_t> pixels;
};
//void initOptix()
//{
//    // -------------------------------------------------------
//    // check for available optix7 capable devices
//    // -------------------------------------------------------
//    cudaFree(0);
//    int numDevices;
//    cudaGetDeviceCount(&numDevices);
//    if (numDevices == 0)
//        throw std::runtime_error("#osc: no CUDA capable devices found!");
//    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;
//
//    // -------------------------------------------------------
//    // initialize optix
//    // -------------------------------------------------------
//    OPTIX_CHECK(optixInit());
//}

extern "C" int main(int ac,char **av) {
    try {
        TriangleMesh model;
        model.addCube(gdt::vec3f(0.f, -1.5f, 0.f), gdt::vec3f(10.f, .1f, 10.f));

        model.addCube(gdt::vec3f(0.f, 0.f, 0.f), gdt::vec3f(2.f, 2.f, 2.f));

        Camera camera = { /*from*/gdt::vec3f(-10.f,2.f,-12.f),
            /* at */gdt::vec3f(0.f,0.f,0.f),
            /* up */gdt::vec3f(0.f,1.f,0.f) };

        // something approximating the scale of the world, so the
        // camera knows how much to move for any given user interaction:
        const float worldScale = 10.f;
        SampleWindow* window = new SampleWindow("Optix 7 Course Example",
            model, camera, worldScale);

        window->run();
    }
    catch (std::runtime_error& e) {
        std::cout << GDT_TERMINAL_RED << "FATAL ERROR:" << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}