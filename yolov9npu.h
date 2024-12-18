//--------------------------------------------------------------------------------------
// yolov4.h
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "DeviceResources.h"
#include "StepTimer.h"
#include "MediaEnginePlayer.h"
#include "ssd_anchors.h"
#include "depixelator.h"


using UniqueNativePtr = std::unique_ptr<void, void (*)(void*)>;

using Size = int64_t;
using String = std::string;
template <typename T>
using Vector = std::vector<T>;

template <typename T>
struct Vec1
{
    const T* data;
    Size x;
    Vec1(const T* data, Size x) : data(data), x(x) {}
    const T& operator[](Size i) const { return data[i]; }
};
template <typename T>
struct Vec2
{
    const T* data;
    Size y, x;
    Vec2(const T* data, Size y, Size x) : data(data), y(y), x(x) {}
    Vec1<T> operator[](Size i) const { return Vec1<T>(data + i * x, x); }
};

template <typename T>
struct Vec3
{
    const T* data;
    Size z, y, x;
    Vec3(const T* data, Size z, Size y, Size x) : data(data), z(z), y(y), x(x) {}
    Vec2<T> operator[](Size i) const { return Vec2<T>(data + i * y * x, y, x); }
};

template <typename T>
struct Vec4
{
    const T* data;
    Size w, z, y, x;
    Vec4(const T* data, Size w, Size z, Size y, Size x) : data(data), w(w), z(z), y(y), x(x) {}
    Vec3<T> operator[](Size i) const { return Vec3<T>(data + i * z * y * x, z, y, x); }
};

struct Detection
{
    /// @brief The x-coordinate of the top-left corner of the bounding box.
    float x;

    /// @brief The y-coordinate of the top-left corner of the bounding box.
    float y;

    /// @brief The width of the bounding box.
    float w;

    /// @brief The height of the bounding box.
    float h;

    /// @brief The index of the detected object, typically corresponding to the class index in the model.
    Size index;

    /// @brief The confidence level of the detection.
    float confidence;
};

using Detections = Vector<Detection>;




namespace YoloV4Constants
{

    
    // The classes of objects that yolov4 can detect
    static const char* const c_classes[] =
    {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "potted plant", "bed", "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    };
    

    static const uint32_t c_numClasses = ARRAYSIZE(c_classes);

    // texture size the input tensor expects
    static  uint32_t t_inputWidth = 1280;
    static  uint32_t t_inputHeight = 720;

    // Input images are rescaled to 512x288 before being fed into the model
    static uint32_t c_inputWidth = 512;
    static uint32_t c_inputHeight = 288;

    // Discard any predictions which have a low score (i.e. predictions that the model isn't very confident about)
    static const float c_scoreThreshold = 0.25f;

    // Threshold for non-maximal suppression (NMS) which determines how much overlap between bounding boxes is needed
    // before they're eliminated
    static const float c_nmsThreshold = 0.213f;

    // YoloV4 produces bounding boxes on different scales (small, medium, large) and the outputs of the model need
    // to be scaled according to their appropriate constants.
    struct BBoxData
    {
        float xyScale;
        float stride;
        std::array<float, 6> anchors;

        static BBoxData Small()
        {
            BBoxData data;
            data.xyScale = 1.2f;
            data.stride = 8;
            data.anchors = { 12,16,   19,36,   40,28   };
            return data;
        }

        static BBoxData Medium()
        {
            BBoxData data;
            data.xyScale = 1.1f;
            data.stride = 16;
            data.anchors = { 36,75,   76,55,   72,146  };
            return data;
        }

        static BBoxData Large()
        {
            BBoxData data;
            data.xyScale = 1.05f;
            data.stride = 32;
            data.anchors = { 142,110, 192,243, 459,401 };
            return data;
        }
    };
};

class SmoothedFPS
{
public:
    SmoothedFPS(float secondsInterval = 1.f)
    {
        Initialize(secondsInterval);
    }

    void Initialize(float secondsInterval = 1.f)
    {
        m_secondsInterval = secondsInterval;
        m_timeAccumulator = 0.0f;
        m_frameAccumulator = 0;
        m_smoothedFPS = 0.0f;
    }

    void Tick(float DeltaTime)
    {
        m_timeAccumulator += DeltaTime;
        ++m_frameAccumulator;

        if (m_timeAccumulator >= m_secondsInterval)
        {
            m_smoothedFPS = (float)m_frameAccumulator / m_timeAccumulator;
            m_timeAccumulator = 0.0f;
            m_frameAccumulator = 0;
        }
    }

    float GetFPS() const { return m_smoothedFPS; }

private:
    float m_smoothedFPS;
    float m_timeAccumulator;
    uint32_t m_frameAccumulator;
    float m_secondsInterval;
};

enum class TensorLayout
{
    Default,
    NHWC
};

struct Prediction
{
    // Bounding box coordinates
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int32_t predictedClass;
    std::vector<std::pair<float, float> > m_keypoints;
    int i, j;
    std::vector<float> mask_weights;
    depixelator::Polylines m_polylines;
};

struct Model_t
{
    std::wstring                                    m_modelfile;
    Ort::Session                                    m_session{ nullptr };
    Ort::Value                                      m_inputTensor{ nullptr };
    Ort::Value                                      m_outputTensor{ nullptr };
    std::vector<int64_t>                            m_inputShape;
    std::vector<int64_t>                            m_outputShape;
    ONNXTensorElementDataType                       m_inputDataType;
    size_t                                          m_inputWidth;
    size_t                                          m_inputHeight;
    std::vector<std::byte>                          m_inputBuffer;
};

// A basic sample implementation that creates a D3D12 device and
// provides a render loop.
class Sample final : public DX::IDeviceNotify
{
public:

    Sample() noexcept(false);
    ~Sample();

    // Initialization and management
    bool Initialize(HWND window, int width, int height, bool run_on_gpu = false);

    // Basic render loop
    void Tick();

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // Messages
    void OnActivated();
    void OnDeactivated();
    void OnSuspending();
    void OnResuming();
    void OnWindowMoved();
    void OnWindowSizeChanged(int width, int height);
    void OnNewFile(const wchar_t* filename);
    void OnNewMopdel(const wchar_t* modelfile, bool bAddModel);

    // Properties
    void GetDefaultSize(int& width, int& height) const;

private:

    void Update(DX::StepTimer const& timer);
    void Render();

    void Clear();

    void CreateDeviceDependentResources();
    void CreateTextureResources();
    void InitializeDirectMLResources(const wchar_t* model_path = nullptr, bool Addmdodel = false);
    void CreateUIResources();
    void CreateWindowSizeDependentResources();

    bool CopySharedVideoTextureTensor(std::vector<std::byte>& inputbuffer, Model_t* model);


    void GetNonGraphicsAdapter(IDXCoreAdapterList* adapterList, IDXCoreAdapter** outAdapter);

    void InitializeDirectML(ID3D12Device1** d3dDeviceOut, ID3D12CommandQueue** commandQueueOut, IDMLDevice** dmlDeviceOut,
        ID3D12CommandAllocator** commandAllocatorOut,
        ID3D12GraphicsCommandList** commandListOut);

    struct ModelOutput
    {
        // DEFAULT buffer containing the output contents
        Microsoft::WRL::ComPtr<ID3D12Resource>      output;

        // READBACK buffer for retrieving the output contents from the GPU
        Microsoft::WRL::ComPtr<ID3D12Resource>      readback;

        // Size, format, etc. of the output data
        dml::TensorDesc                             desc;
    };

    // Given a raw output of the model, retrieves the predictions (a bounding box, detected class, and score) of the
    // model.

    // objects
    void GetPredictions(const std::byte* outputData, std::vector<int64_t> & shape, const  std::vector<std::string>& output_names, Model_t* model);
    void GetPredictions(std::vector<const std::byte*>& outputData, std::vector<std::vector<int64_t>>& shapes, const  std::vector<std::string>& output_names, Model_t* model);
    void GetPredictions2(std::vector<const std::byte*>& outputData, std::vector<std::vector<int64_t>>& shapes, const  std::vector<std::string>& output_names, Model_t* model);

    // faces
    void GetFaces(std::vector<const std::byte*>& outputData, std::vector<std::vector<int64_t>>& shapes, Model_t* model);
    typedef std::vector<onnxmediapipe::Anchor> Anchors;
    std::vector<Anchors> m_anchors;

    // image
    void GetImage(const std::byte* outputData, std::vector<int64_t>& shape, Model_t* model, ONNXTensorElementDataType datatype);
    void GetMask(const std::byte* outputData, std::vector<int64_t>& shape, Model_t* model, ONNXTensorElementDataType datatype);
   
    bool LoadTextureFromMemory(const std::byte* image_data, uint32_t width, uint32_t height, ID3D12Device* d3d_device, D3D12_CPU_DESCRIPTOR_HANDLE srv_cpu_handle, ID3D12Resource** out_tex_resource);
    void NewTexture(const uint8_t * image_data, uint32_t width, uint32_t height);
    // 
    // 
    bool                                            m_run_on_gpu;
    // 
    // Device resources
    std::unique_ptr<DX::DeviceResources>            m_deviceResources;

    // Rendering loop timer
    DX::StepTimer                                   m_timer;

    // Input devices
    std::unique_ptr<DirectX::GamePad>               m_gamePad;
    std::unique_ptr<DirectX::Keyboard>              m_keyboard;
    DirectX::GamePad::ButtonStateTracker            m_gamePadButtons;
    DirectX::Keyboard::KeyboardStateTracker         m_keyboardButtons;
    bool                                            m_ctrlConnected;

    // DirectXTK objects
    std::unique_ptr<DirectX::GraphicsMemory>        m_graphicsMemory;
    std::unique_ptr<DirectX::DescriptorHeap>        m_SRVDescriptorHeap;

    // UI
    SmoothedFPS                                     m_fps;
    std::unique_ptr<DirectX::BasicEffect>           m_lineEffect;
    std::unique_ptr<DirectX::BasicEffect>           m_lineEffect2;
    std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>> m_lineBatch;
    std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionColor>> m_lineBatch2;
    std::unique_ptr<DirectX::DescriptorHeap>        m_fontDescriptorHeap;
    std::unique_ptr<DirectX::SpriteBatch>           m_spriteBatch;
    std::unique_ptr<DirectX::SpriteFont>            m_labelFont;
    std::unique_ptr<DirectX::SpriteFont>            m_labelFontBold;
    std::unique_ptr<DirectX::SpriteFont>            m_legendFont;
    std::unique_ptr<DirectX::SpriteFont>            m_ctrlFont;

    // Video player
public:
    std::unique_ptr<MediaEnginePlayer>              m_player;
private:
    HANDLE                                          m_sharedVideoTexture;
    LONGLONG                                        m_pts = 0;

    // Direct3D 12 objects for rendering texture to screen
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_texRootSignatureNN;           // Nearest-neighbor texture upscale
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_texPipelineStateNN;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_texRootSignatureLinear;       // Bilinear texture upscale
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_texPipelineStateLinear;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_tensorRenderRootSignature;    // Render from DML tensor format to texture
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_tensorRenderPipelineState;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_texture;                      // Static input texture to render, if USE_VIDEO == 0
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_videoTexture;                 // Input video frame to render, if USE_VIDEO == 1
    uint32_t                                        m_origTextureHeight;
    uint32_t                                        m_origTextureWidth;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_vertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW                        m_vertexBufferView;
    D3D12_INDEX_BUFFER_VIEW                         m_indexBufferView;

    // Compute objects for converting texture to DML tensor format
    Microsoft::WRL::ComPtr<ID3D12PipelineState>     m_computePSO;
    Microsoft::WRL::ComPtr<ID3D12RootSignature>     m_computeRootSignature;

    // DirectML objects
    std::wstring                                    m_device_name;
    Microsoft::WRL::ComPtr<IDMLDevice>              m_dmlDevice;
    Microsoft::WRL::ComPtr<IDMLCommandRecorder>     m_dmlCommandRecorder;


    ComPtr<ID3D12Device1>                           m_d3dDevice;
    ComPtr<ID3D12CommandQueue>                      m_commandQueue;
    ComPtr<ID3D12CommandAllocator>                  m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList>               m_commandList;

    // Shared Resources
    std::unique_ptr<DirectX::DescriptorHeap>        m_dmlDescriptorHeap;

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelInput;
    ModelOutput                                     m_modelSOutput;
    ModelOutput                                     m_modelMOutput;
    ModelOutput                                     m_modelLOutput;

    

    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelPersistentResource;
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_modelTemporaryResource;

    // DirectMLX operations
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>    m_dmlGraph;
    Microsoft::WRL::ComPtr<IDMLBindingTable>        m_dmlBindingTable;
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_dmlOpInitializer;


    const OrtDmlApi* m_ortDmlApi{ nullptr };

    std::vector<std::unique_ptr<Model_t>>           m_models;

    DWORD                                           m_dxgiFactoryFlags;

    Vector<Detections>                              m_detections;
    std::vector<Prediction>                         m_preds;

    std::vector<byte>                               m_Out;
    std::chrono::duration<double, std::milli>        m_copypixels_tensor_duration;
    std::chrono::duration<double, std::milli>        m_inference_duration;
    std::chrono::duration<double, std::milli>        m_output_duration;
    
    Microsoft::WRL::ComPtr<ID2D1Factory7>           m_d2d1_factory;
    Microsoft::WRL::ComPtr<ID2D1Device6>            m_d2d1_device;
    Microsoft::WRL::ComPtr<ID2D1DeviceContext>      m_d2dContext;

    std::unique_ptr<DirectX::SpriteBatch>           m_sprite; // output tensor image
    Microsoft::WRL::ComPtr<ID3D12Resource>          m_output_texture;
    std::vector<uint8_t>                            m_mask;
    std::vector<uint8_t>                            m_pred_mask;
    bool                                            m_mask_ready = false;
    int                                             m_mask_width;
    int                                             m_mask_height;
    //wil::unique_handle                              m_fenceEvent{ nullptr };

    // DirectX index enums
    enum SrvDescriptors : uint32_t
    {
        e_descTexture,
        e_descModelInput,
        e_outputTensor,
        e_srvDescCount
    };

    enum FontDescriptors : uint32_t
    {
        e_descLabelFont,
        e_descLabelFontBold,
        e_descLegendFont,
        e_descCtrlFont,
        e_fontDescCount,
    };

    enum ComputeRootParameters : uint32_t
    {
        e_crpIdxCB = 0,
        e_crpIdxSRV,
        e_crpIdxUAV,
        e_crpIdxCount
    };

    enum TensorRenderRootParameters : uint32_t
    {
        e_rrpIdxCB = 0,
        e_rrpIdxSRV,
        e_rrpIdxCount
    };
};