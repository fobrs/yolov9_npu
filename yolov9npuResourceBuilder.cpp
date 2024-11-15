#include "pch.h"

#include "yolov9npu.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"


#include "TensorHelper.h"

using Microsoft::WRL::ComPtr;

using namespace DirectX;

static bool TryGetProperty(IDXCoreAdapter* adapter, DXCoreAdapterProperty prop, std::string& outputValue)
{
    if (adapter->IsPropertySupported(prop))
    {
        size_t propSize;
        THROW_IF_FAILED(adapter->GetPropertySize(prop, &propSize));

        outputValue.resize(propSize);
        THROW_IF_FAILED(adapter->GetProperty(prop, propSize, outputValue.data()));

        // Trim any trailing nul characters. 
        while (!outputValue.empty() && outputValue.back() == '\0')
        {
            outputValue.pop_back();
        }

        return true;
    }
    return false;
}

static void GetNonGraphicsAdapter(IDXCoreAdapterList* adapterList, IDXCoreAdapter** outAdapter)
{
    for (uint32_t i = 0, adapterCount = adapterList->GetAdapterCount(); i < adapterCount; i++)
    {
        ComPtr<IDXCoreAdapter> possibleAdapter;
        THROW_IF_FAILED(adapterList->GetAdapter(i, IID_PPV_ARGS(&possibleAdapter)));

        if (!possibleAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS))
        {
            *outAdapter = possibleAdapter.Detach();
            return;
        }
    }
    *outAdapter = nullptr;
}

void Sample::InitializeDirectML(ID3D12Device1** d3dDeviceOut, ID3D12CommandQueue** commandQueueOut, IDMLDevice** dmlDeviceOut,
    ID3D12CommandAllocator** commandAllocatorOut,
    ID3D12GraphicsCommandList** commandListOut)
{
#if 0
    // is extermely slow when createing the ort::Session
#if defined(_DEBUG)
    // Enable the debug layer (requires the Graphics Tools "optional feature").
    //
    // NOTE: Enabling the debug layer after device creation will invalidate the active device.
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf()))))
        {
            debugController->EnableDebugLayer();
        }
        else
        {
            OutputDebugStringA("WARNING: Direct3D Debug Device is not available\n");
        }

        ComPtr<IDXGIInfoQueue> dxgiInfoQueue;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(dxgiInfoQueue.GetAddressOf()))))
        {
            m_dxgiFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;

            dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, true);
            dxgiInfoQueue->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, true);

            DXGI_INFO_QUEUE_MESSAGE_ID hide[] =
            {
                80 /* IDXGISwapChain::GetContainingOutput: The swapchain's adapter does not control the output on which the swapchain's window resides. */,
            };
            DXGI_INFO_QUEUE_FILTER filter = {};
            filter.DenyList.NumIDs = _countof(hide);
            filter.DenyList.pIDList = hide;
            dxgiInfoQueue->AddStorageFilterEntries(DXGI_DEBUG_DXGI, &filter);
        }
    }
#endif
#endif

    // Create Adapter Factory
    ComPtr<IDXCoreAdapterFactory> factory;

    // Note: this module is not currently properly freed. Outside of sample usage, this module should freed e.g. with an explicit free or through wil::unique_hmodule.
    HMODULE dxCoreModule = LoadLibraryW(L"DXCore.dll");

    if (dxCoreModule)
    {
        auto dxcoreCreateAdapterFactory = reinterpret_cast<HRESULT(WINAPI*)(REFIID, void**)>(
            GetProcAddress(dxCoreModule, "DXCoreCreateAdapterFactory")
            );
        if (dxcoreCreateAdapterFactory)
        {
            dxcoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
        }
    }

    // Create the DXCore Adapter, for the purposes of selecting NPU we look for (!GRAPHICS && (GENERIC_ML || CORE_COMPUTE))
    ComPtr<IDXCoreAdapter> adapter;
    ComPtr<IDXCoreAdapterList> adapterList;
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_1_0_GENERIC;

    if (factory)
    {
        THROW_IF_FAILED(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, IID_PPV_ARGS(&adapterList)));

        if (adapterList->GetAdapterCount() > 0)
        {
            GetNonGraphicsAdapter(adapterList.Get(), adapter.GetAddressOf());
        }

        if (!adapter)
        {
            featureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
            THROW_IF_FAILED(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, IID_PPV_ARGS(&adapterList)));
            GetNonGraphicsAdapter(adapterList.Get(), adapter.GetAddressOf());
        }
    }

    if (adapter)
    {
        std::string adapterName;
        if (TryGetProperty(adapter.Get(), DXCoreAdapterProperty::DriverDescription, adapterName))
        {
            printf("Successfully found adapter %s\n", adapterName.c_str());
        }
        else
        {
            printf("Failed to get adapter description.\n");
        }
    }

    // Create the D3D12 Device
    ComPtr<ID3D12Device1> d3dDevice;
    if (adapter)
    {
        // Note: this module is not currently properly freed. Outside of sample usage, this module should freed e.g. with an explicit free or through wil::unique_hmodule.
        HMODULE d3d12Module = LoadLibraryW(L"d3d12.dll");
        if (d3d12Module)
        {
            auto d3d12CreateDevice = reinterpret_cast<HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void*)>(
                GetProcAddress(d3d12Module, "D3D12CreateDevice")
                );
            if (d3d12CreateDevice)
            {
                // The GENERIC feature level minimum allows for the creation of both compute only and generic ML devices.
                THROW_IF_FAILED(d3d12CreateDevice(adapter.Get(), featureLevel, IID_PPV_ARGS(&d3dDevice)));
            }
        }
    }

    // Create the DML Device and D3D12 Command Queue
    ComPtr<IDMLDevice> dmlDevice;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    if (d3dDevice)
    {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        THROW_IF_FAILED(d3dDevice->CreateCommandQueue(
            &queueDesc,
            IID_PPV_ARGS(commandQueue.ReleaseAndGetAddressOf())));



        THROW_IF_FAILED(d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(commandAllocator.ReleaseAndGetAddressOf())));
        THROW_IF_FAILED(d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, commandAllocator.Get(), nullptr, IID_PPV_ARGS(commandList.ReleaseAndGetAddressOf())));


        // Note: this module is not currently properly freed. Outside of sample usage, this module should freed e.g. with an explicit free or through wil::unique_hmodule.
        HMODULE dmlModule = LoadLibraryW(L"DirectML.dll");
        if (dmlModule)
        {
            auto dmlCreateDevice = reinterpret_cast<HRESULT(WINAPI*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, DML_FEATURE_LEVEL, REFIID, void*)>(
                GetProcAddress(dmlModule, "DMLCreateDevice1")
                );
            if (dmlCreateDevice)
            {
                THROW_IF_FAILED(dmlCreateDevice(d3dDevice.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(dmlDevice.ReleaseAndGetAddressOf())));
            }
        }
    }

    d3dDevice.CopyTo(d3dDeviceOut);
    commandQueue.CopyTo(commandQueueOut);
    dmlDevice.CopyTo(dmlDeviceOut);
    commandAllocator.CopyTo(commandAllocatorOut);
    commandList.CopyTo(commandListOut);
}


void Sample::InitializeDirectMLResources(wchar_t * model_path)
{
    const OrtApi& ortApi = Ort::GetApi();
    static Ort::Env s_OrtEnv{ nullptr };
    s_OrtEnv = Ort::Env(Ort::ThreadingOptions{});
    s_OrtEnv.DisableTelemetryEvents();

    auto sessionOptions = Ort::SessionOptions{};
    sessionOptions.DisableMemPattern();
    sessionOptions.DisablePerSessionThreads();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    //sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    //sessionOptions.AddConfigEntry("session.load_model_format", "ORT");
    //sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    m_ortDmlApi = nullptr;
    Ort::ThrowOnError(ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&m_ortDmlApi)));
    Ort::ThrowOnError(m_ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, m_dmlDevice.Get(), m_commandQueue.Get()));

    try
    {
        if (model_path == nullptr)
        {

            // Create the session

            //auto session = Ort::Session(s_OrtEnv, L"mobilenetv2-7-fp16.onnx", sessionOptions);
            // model from here:
            // https://github.com/DakeQQ/YOLO-Depth-Estimation-for-Android

            //wchar_t * modelfile = L"Model_Yolo_v9c.ort";
            //wchar_t* modelfile = L"Model_Yolo_v9c_f16.onnx";
            //wchar_t* modelfile = L"Model_Yolo_v9c_f16_h1088_w1920.onnx";
            //wchar_t* modelfile = L"yolov11_det.onnx";
            //wchar_t* modelfile = L"yolo11n.onnx";
            //wchar_t* modelfile = L"yolov10m.onnx";
            wchar_t* modelfile = L"yolov8_det.onnx";

            m_modelfile = std::wstring(modelfile);

            std::wstring model_path = L".\\Data\\" + m_modelfile;
            m_session = Ort::Session(s_OrtEnv, model_path.c_str(), sessionOptions);
        }
        else
        {
            const wchar_t* pstrName = wcsrchr(model_path, '\\');
            if (!pstrName)
            {
                m_modelfile = std::wstring(model_path);
            }
            else
            {
                pstrName++;
                m_modelfile = std::wstring(pstrName);
            }

            m_session = Ort::Session(s_OrtEnv, model_path, sessionOptions);
        }
        
    }
    catch (const std::runtime_error& re) {
        std::cerr << "Runtime error: " << re.what() << std::endl;
        exit(1);
    }
    catch (const std::exception& ex)
    {
        const char* err = ex.what();
        MessageBoxA(0, err, "Error loading model", MB_YESNO);
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        exit(1);
    }
  
    //const char* inputName = "images";
    //const char* outputName = "output0";

    // Create input tensor
    Ort::TypeInfo type_info = m_session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input = CreateDmlValue(tensor_info, m_commandQueue.Get());
    m_inputTensor = std::move(input.first);

    //const auto memoryInfo = inputTensor.GetTensorMemoryInfo();
    Ort::MemoryInfo memoryInfo0 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const auto memoryInfo = m_inputTensor.GetTensorMemoryInfo();
    Ort::Allocator allocator0(m_session, memoryInfo0);
    Ort::Allocator allocator(m_session, memoryInfo);

    auto meta = m_session.GetModelMetadata();
    auto modelname = meta.GetGraphNameAllocated(allocator0);

    auto inputName = m_session.GetInputNameAllocated(0, allocator0);
    auto inputTypeInfo = m_session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    m_inputShape = inputTensorInfo.GetShape();
    m_inputDataType = inputTensorInfo.GetElementType();
    if (m_inputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && m_inputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        throw std::invalid_argument("Model input must be of type float32 or float16");
    }
    if (m_inputShape.size() < 3)
    {
        throw std::invalid_argument("Model input must be an image with 3 or 4 dimensions");
    }

    const size_t inputChannels = m_inputShape[m_inputShape.size() - 3];
    const size_t inputHeight = m_inputShape[m_inputShape.size() - 2];
    const size_t inputWidth = m_inputShape[m_inputShape.size() - 1];


    m_inputWidth = inputWidth;
    m_inputHeight = inputHeight;

    const size_t inputElementSize = m_inputDataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);

    auto outputName = m_session.GetOutputNameAllocated(0, allocator0);
    auto tensors = m_session.GetOutputCount();
    auto outputTypeInfo = m_session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    m_outputShape = outputTensorInfo.GetShape();
    auto outputDataType = outputTensorInfo.GetElementType();
    if (outputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && outputDataType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        throw std::invalid_argument("Model output must be of type float32 or float16");
    }
    if (m_outputShape.size() < 3)
    {
        throw std::invalid_argument("Model output must be an image with 3 or 4 dimensions");
    }

    const size_t outputChannels = m_outputShape[m_outputShape.size() - 3];
    const size_t outputHeight = m_outputShape[m_outputShape.size() - 2];
    const size_t outputWidth = m_outputShape[m_outputShape.size() - 1];
    const size_t outputElementSize = outputDataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);   
}
