#include "pch.h"
#include "Resizer.h"
#include "ScreenGrab.h"
#include "utilities.h"
//#include <wincodec.h>

using namespace DirectX;

Resizer::Resizer() : 
	m_Device(nullptr),
	m_DeviceContext(nullptr),
	m_SamplerLinear(nullptr),
	m_BlendState(nullptr),
	m_VertexShader(nullptr),
	m_PixelShader(nullptr),
	m_InputLayout(nullptr),
	m_TargetTexture(nullptr),
    m_RTV(nullptr),
    m_SrcTexture(nullptr),
    m_SrcSrv(nullptr)
{
}

Resizer::~Resizer()
{
    CleanRefs();
}

HRESULT Resizer::InitDx()
{
    HRESULT hr;

    // Driver types supported
    D3D_DRIVER_TYPE DriverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };
    UINT NumDriverTypes = ARRAYSIZE(DriverTypes);

    // Feature levels supported
    D3D_FEATURE_LEVEL FeatureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_1
    };
    UINT NumFeatureLevels = ARRAYSIZE(FeatureLevels);
    D3D_FEATURE_LEVEL FeatureLevel;

    // Create device
    for (UINT DriverTypeIndex = 0; DriverTypeIndex < NumDriverTypes; ++DriverTypeIndex)
    {
        hr = D3D11CreateDevice(nullptr, DriverTypes[DriverTypeIndex], nullptr, 0, FeatureLevels, NumFeatureLevels,
            D3D11_SDK_VERSION, &m_Device, &FeatureLevel, &m_DeviceContext);
        if (SUCCEEDED(hr))
        {
            // Device creation succeeded, no need to loop anymore
            break;
        }
    }
    return hr;
}

HRESULT Resizer::Prepare(SIZE targetSize) 
{
    HRESULT hr;

    // Create target texture
    D3D11_TEXTURE2D_DESC targetDesc;
    InitializeDesc(targetSize, &targetDesc);
    hr = m_Device->CreateTexture2D(&targetDesc, nullptr, &m_TargetTexture);

    // Make new render target view
    hr = MakeRTV();
    if (!SUCCEEDED(hr))
        return hr;

    // Set view port
    SetViewPort(targetSize);

    //Clear our render target and depth/stencil view
    float bgColor[4] = { 1.0f, 0.1f, 0.1f, 0.0f };
    m_DeviceContext->ClearRenderTargetView(m_RTV, bgColor);

    // Create the sample state
    D3D11_SAMPLER_DESC SampDesc;
    RtlZeroMemory(&SampDesc, sizeof(SampDesc));
    SampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    SampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    SampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    SampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    SampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    SampDesc.MinLOD = 0.0f;
    SampDesc.MaxLOD = 0.0f;
    hr = m_Device->CreateSamplerState(&SampDesc, &m_SamplerLinear);
    if (!SUCCEEDED(hr))
        return hr;

    // Create the blend state
    D3D11_BLEND_DESC BlendStateDesc;
    BlendStateDesc.AlphaToCoverageEnable = FALSE;
    BlendStateDesc.IndependentBlendEnable = FALSE;
    BlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    BlendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    BlendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    BlendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    BlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    BlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    BlendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    BlendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = m_Device->CreateBlendState(&BlendStateDesc, &m_BlendState);
    if (!SUCCEEDED(hr))
        return hr;

    // Initialize shaders
    hr = InitShaders();
    if (!SUCCEEDED(hr))
        return hr;

    return S_OK;
}

HRESULT Resizer::Draw()
{
    HRESULT hr;

    // Vertices for drawing whole texture
    VERTEX Vertices[NUMVERTICES] =
    {
        {XMFLOAT3(-1.0f, -1.0f, 0), XMFLOAT2(0.0f, 1.0f)},
        {XMFLOAT3(-1.0f, 1.0f, 0), XMFLOAT2(0.0f, 0.0f)},
        {XMFLOAT3(1.0f, -1.0f, 0), XMFLOAT2(1.0f, 1.0f)},
        {XMFLOAT3(1.0f, 1.0f, 0), XMFLOAT2(1.0f, 0.0f)},
    };

    // Set resources
    UINT Stride = sizeof(VERTEX);
    UINT Offset = 0;
    FLOAT blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
    m_DeviceContext->OMSetBlendState(nullptr, blendFactor, 0xffffffff);
    m_DeviceContext->OMSetRenderTargets(1, &m_RTV, nullptr);
    m_DeviceContext->VSSetShader(m_VertexShader, nullptr, 0);
    m_DeviceContext->PSSetShader(m_PixelShader, nullptr, 0);
    m_DeviceContext->PSSetShaderResources(0, 1, &m_SrcSrv);
    m_DeviceContext->PSSetSamplers(0, 1, &m_SamplerLinear);
    m_DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    D3D11_BUFFER_DESC BufferDesc;
    RtlZeroMemory(&BufferDesc, sizeof(BufferDesc));
    BufferDesc.Usage = D3D11_USAGE_DEFAULT;
    BufferDesc.ByteWidth = sizeof(VERTEX) * NUMVERTICES;
    BufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    BufferDesc.CPUAccessFlags = 0;
    D3D11_SUBRESOURCE_DATA InitData;
    RtlZeroMemory(&InitData, sizeof(InitData));
    InitData.pSysMem = Vertices;

    ID3D11Buffer* VertexBuffer = nullptr;

    // Create vertex buffer
    hr = m_Device->CreateBuffer(&BufferDesc, &InitData, &VertexBuffer);
    if (FAILED(hr))
    {
        if (m_SrcSrv)
        {
            m_SrcSrv->Release();
            m_SrcSrv = nullptr;
        }
        return S_FALSE;
    }
    m_DeviceContext->IASetVertexBuffers(0, 1, &VertexBuffer, &Stride, &Offset);

    //m_DeviceContext->PSSetShaderResources(0, 1, &source);


    // Draw textured quad onto render target
    m_DeviceContext->Draw(NUMVERTICES, 0);

    m_DeviceContext->Flush();

    VertexBuffer->Release();
    VertexBuffer = nullptr;

    // Release shader resource
    if (m_SrcSrv)
    {
        m_SrcSrv->Release();
        m_SrcSrv = nullptr;
    }

    return S_OK;
}


HRESULT Resizer::LoadTexture(ID3D11Resource * src)
{
    //  return DirectX::CreateWICTextureFromFile(m_Device, path.c_str(), &m_SrcTexture, &m_SrcSrv);
    m_SrcTexture = src;
    return S_OK;
}

HRESULT Resizer::GetTargetTexture(ID3D11Texture2D** target)
{
    //  return DirectX::CreateWICTextureFromFile(m_Device, path.c_str(), &m_SrcTexture, &m_SrcSrv);
    if (target)
        *target = m_TargetTexture;
    return S_OK;
}


HRESULT Resizer::ReadFile(const std::wstring& path) 
{
 //  return DirectX::CreateWICTextureFromFile(m_Device, path.c_str(), &m_SrcTexture, &m_SrcSrv);
    return S_OK;
}
HRESULT Resizer::SaveFile(const std::wstring& path)
{
    return DirectX::SaveDDSTextureToFile(m_DeviceContext, m_TargetTexture, path.c_str());
  // return S_OK;
}

HRESULT Resizer::InitializeDesc(_In_ SIZE size, _Out_ D3D11_TEXTURE2D_DESC* pTargetDesc)
{
    // Create shared texture for the target view
    RtlZeroMemory(pTargetDesc, sizeof(D3D11_TEXTURE2D_DESC));
    pTargetDesc->Width = size.cx;
    pTargetDesc->Height = size.cy;
    pTargetDesc->MipLevels = 1;
    pTargetDesc->ArraySize = 1;
    pTargetDesc->Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    pTargetDesc->SampleDesc.Count = 1;
    pTargetDesc->Usage = D3D11_USAGE_DEFAULT;
    pTargetDesc->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    pTargetDesc->CPUAccessFlags = 0;
    pTargetDesc->MiscFlags = 0;

    return S_OK;
}
HRESULT Resizer::MakeRTV()
{
    HRESULT hr = m_Device->CreateRenderTargetView(m_TargetTexture, nullptr, &m_RTV);
    RETURN_ON_BAD_HR(hr);

    m_DeviceContext->OMSetRenderTargets(1, &m_RTV, nullptr);

    return S_OK;
}
void Resizer::SetViewPort(SIZE size)
{
    D3D11_VIEWPORT VP = {};
    VP.Width = static_cast<FLOAT>(size.cx);
    VP.Height = static_cast<FLOAT>(size.cy);
    VP.MinDepth = 0.0f;
    VP.MaxDepth = D3D11_MAX_DEPTH;
    VP.TopLeftX = 0;
    VP.TopLeftY = 0;
    m_DeviceContext->RSSetViewports(1, &VP);
}
//
// Initialize shaders for drawing to screen
//
HRESULT Resizer::InitShaders()
{
    HRESULT hr;

    UINT Size = ARRAYSIZE(g_VS);
    hr = m_Device->CreateVertexShader(g_VS, Size, nullptr, &m_VertexShader);
    RETURN_ON_BAD_HR(hr);

    D3D11_INPUT_ELEMENT_DESC Layout[] =
    {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };
    UINT NumElements = ARRAYSIZE(Layout);
    hr = m_Device->CreateInputLayout(Layout, NumElements, g_VS, Size, &m_InputLayout);
    RETURN_ON_BAD_HR(hr);

    m_DeviceContext->IASetInputLayout(m_InputLayout);

    Size = ARRAYSIZE(g_PS);
    hr = m_Device->CreatePixelShader(g_PS, Size, nullptr, &m_PixelShader);
    RETURN_ON_BAD_HR(hr);

    return S_OK;
}
//
// Releases all references
//
void Resizer::CleanRefs()
{
    if (m_VertexShader)
    {
        m_VertexShader->Release();
        m_VertexShader = nullptr;
    }

    if (m_PixelShader)
    {
        m_PixelShader->Release();
        m_PixelShader = nullptr;
    }

    if (m_InputLayout)
    {
        m_InputLayout->Release();
        m_InputLayout = nullptr;
    }

    if (m_RTV)
    {
        m_RTV->Release();
        m_RTV = nullptr;
    }

    if (m_SamplerLinear)
    {
        m_SamplerLinear->Release();
        m_SamplerLinear = nullptr;
    }

    if (m_BlendState)
    {
        m_BlendState->Release();
        m_BlendState = nullptr;
    }

    if (m_TargetTexture)
    {
        m_TargetTexture->Release();
        m_TargetTexture = nullptr;
    }

    if (m_SrcTexture)
    {
        //m_SrcTexture->Release();
        m_SrcTexture = nullptr;
    }

    if (m_SrcSrv)
    {
        m_SrcSrv->Release();
        m_SrcSrv = nullptr;
    }

    if (m_DeviceContext)
    {
        m_DeviceContext->Release();
        m_DeviceContext = nullptr;
    }

    if (m_Device)
    {
        m_Device->Release();
        m_Device = nullptr;
    }
}
