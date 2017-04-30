/*!
 * @file 
 * @brief This file contains implemenation of phong vertex and fragment shader.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include<math.h>
#include<assert.h>
#include <memory.h>

#include"student/student_shader.h"
#include"student/gpu.h"
#include"student/uniforms.h"
#include "mouseCamera.h"
#include "vertexPuller.h"

/// \addtogroup shader_side Úkoly v shaderech
/// @{

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void phong_vertexShader(
    GPUVertexShaderOutput     *const output,
    GPUVertexShaderInput const*const input ,
    GPU                        const gpu   ){
  /// \todo HOTOVO Naimplementujte vertex shader, který transformuje vstupní vrcholy do clip-space.<br>
    /// <b>Vstupy:</b><br>
    /// Vstupní vrchol by měl v nultém atributu obsahovat pozici vrcholu ve world-space (vec3) a v prvním
    /// atributu obsahovat normálu vrcholu ve world-space (vec3).<br>
    /// <b>Výstupy:</b><br>
    /// Výstupní vrchol by měl v nultém atributu obsahovat pozici vrcholu (vec3) ve world-space a v prvním
    /// atributu obsahovat normálu vrcholu ve world-space (vec3).
    /// Výstupní vrchol obsahuje pozici a normálu vrcholu proto, že chceme počítat osvětlení ve world-space ve fragment shaderu.<br>
  /// <b>Uniformy:</b><br>
  /// Vertex shader by měl pro transformaci využít uniformní proměnné obsahující view a projekční matici.
  /// View matici čtěte z uniformní proměnné "viewMatrix" a projekční matici čtěte z uniformní proměnné "projectionMatrix".
  /// Zachovejte jména uniformních proměnných a pozice vstupních a výstupních atributů.
  /// Pokud tak neučiníte, akceptační testy selžou.<br>
  /// <br>
  /// Využijte vektorové a maticové funkce.
  /// Nepředávajte si data do shaderu pomocí globálních proměnných.
  /// Pro získání dat atributů použijte příslušné funkce vs_interpret* definované v souboru program.h.
  /// Pro získání dat uniformních proměnných použijte příslušné funkce shader_interpretUniform* definované v souboru program.h.
  /// Vrchol v clip-space by měl být zapsán do proměnné gl_Position ve výstupní struktuře.<br>
  /// <b>Seznam funkcí, které jistě použijete</b>:
  ///  - gpu_getUniformsHandle()                  XXX
  ///  - getUniformLocation()                     XXX
  ///  - shader_interpretUniformAsMat4()          XXX
  ///  - vs_interpretInputVertexAttributeAsVec3() XXX
  ///  - vs_interpretOutputVertexAttributeAsVec3()


  Mat4 projectionMatrix = *shader_interpretUniformAsMat4(gpu_getUniformsHandle(gpu), getUniformLocation(gpu, "projectionMatrix"));
  Mat4 viewMatrix = *shader_interpretUniformAsMat4(gpu_getUniformsHandle(gpu), getUniformLocation(gpu, "viewMatrix"));

  Mat4 mult;
  multiply_Mat4_Mat4(&mult, &projectionMatrix, &viewMatrix);

  Vec4 pos;
  Vec3 pos3 = *vs_interpretInputVertexAttributeAsVec3(gpu, input, 0);

  copy_Vec3Float_To_Vec4(&pos, &pos3, 1.f);

  multiply_Mat4_Vec4(&output->gl_Position, &mult, &pos);

  Vec3 *const position = vs_interpretOutputVertexAttributeAsVec3(gpu, output, 0);
  init_Vec3(position, pos3.data[0], pos3.data[1], pos3.data[2]);

  Vec3 vecNormal = *vs_interpretInputVertexAttributeAsVec3(gpu, input, 1);//index of vertex attribute
  Vec3 *const normal = vs_interpretOutputVertexAttributeAsVec3(gpu, output, 1);//index of vertex attribute
  init_Vec3(normal, vecNormal.data[0], vecNormal.data[1], vecNormal.data[2]);

  (void)output;
  (void)input;
  (void)gpu;
}

void phong_fragmentShader(
    GPUFragmentShaderOutput     *const output,
    GPUFragmentShaderInput const*const input ,
    GPU                          const gpu   ){
  /// \todo HOTOVO Naimplementujte fragment shader, který počítá phongův osvětlovací model s phongovým stínováním.<br>
  /// <b>Vstup:</b><br>
  /// Vstupní fragment by měl v nultém fragment atributu obsahovat interpolovanou pozici ve world-space a v prvním
  /// fragment atributu obsahovat interpolovanou normálu ve world-space.<br>
  /// <b>Výstup:</b><br> 
  /// Barvu zapište do proměnné color ve výstupní struktuře.<br>
  /// <b>Uniformy:</b><br>
  /// Pozici kamery přečtěte z uniformní proměnné "cameraPosition" a pozici světla přečtěte z uniformní proměnné "lightPosition".
  /// Zachovejte jména uniformních proměnný.
  /// Pokud tak neučiníte, akceptační testy selžou.<br>
  /// <br>
  /// Dejte si pozor na velikost normálového vektoru, při lineární interpolaci v rasterizaci může dojít ke zkrácení.
  /// Zapište barvu do proměnné color ve výstupní struktuře.
  /// Shininess faktor nastavte na 40.f
  /// Difuzní barvu materiálu nastavte na čistou zelenou.
  /// Spekulární barvu materiálu nastavte na čistou bílou.
  /// Barvu světla nastavte na bílou.
  /// Nepoužívejte ambientní světlo.<br>
  /// <b>Seznam funkcí, které jistě využijete</b>:
  ///  - shader_interpretUniformAsVec3()
  ///  - fs_interpretInputAttributeAsVec3()

  Vec3 cameraPosition;
  Vec3 lightVector;
  sub_Vec3(&lightVector,    shader_interpretUniformAsVec3(gpu_getUniformsHandle(gpu), getUniformLocation(gpu, "lightPosition")),  fs_interpretInputAttributeAsVec3(gpu, input, 0));
  sub_Vec3(&cameraPosition, shader_interpretUniformAsVec3(gpu_getUniformsHandle(gpu), getUniformLocation(gpu, "cameraPosition")), fs_interpretInputAttributeAsVec3(gpu, input, 0));
  normalize_Vec3(&lightVector, &lightVector);
  normalize_Vec3(&cameraPosition, &cameraPosition);

  Vec3 green;
  init_Vec3(&green, 0, 1, 0);

  Vec3 white;
  init_Vec3(&white, 0, 0, 1);

  Vec3 normal;
  normalize_Vec3(&normal, fs_interpretInputAttributeAsVec3(gpu, input, 1));

  Vec3 tmp;
  multiply_Vec3_Float(&tmp, &normal, 2 * MAX(dot_Vec3(&normal, &lightVector),0));
//  normalize_Vec3(&tmp, &tmp);// do not do this, idiot!
  Vec3 reflection;
  sub_Vec3(&reflection, &tmp, &lightVector);
  normalize_Vec3(&reflection, &reflection);

  Vec3 colorDiffuse;
  multiply_Vec3_Float(&colorDiffuse, &green, MAX(dot_Vec3(&normal, &lightVector),0));

  Vec3 colorSpecular;
  multiply_Vec3_Float(&colorSpecular, &white, powf(MAX(dot_Vec3(&cameraPosition, &reflection), 0), 40.f));

  Vec3 color;
  add_Vec3(&color, &colorDiffuse, &colorSpecular);

  init_Vec4(&output->color, color.data[0], color.data[1], color.data[2], 1);

  (void)output;
  (void)input;
  (void)gpu;

}

/// @}
