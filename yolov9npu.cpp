//--------------------------------------------------------------------------------------
// yolov4.cpp
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "pch.h"

#include "yolov9npu.h"

#include "ATGColors.h"
#include "ControllerFont.h"
#include "FindMedia.h"
#include "ReadData.h"

#include "TensorExtents.h"
#include "TensorHelper.h"

double threshold = .40;

const wchar_t* c_videoPath = L"grca-grand-canyon-association-park-store_1280x720.mp4";
const wchar_t* c_imagePath = L"grca-BA-bike-shop_1280x720.jpg";

extern void ExitSample();

using namespace DirectX;

using Microsoft::WRL::ComPtr;

namespace
{
    struct Vertex
    {
        XMFLOAT4 position;
        XMFLOAT2 texcoord;
    };

    int colors[] = { 0xf0fafa, 0x3588d1, 0x4ad59b, 0x399283, 0x97f989, 0x57b230, 0xd8e9b2, 0xff1c5d, 0xf1bb99, 0xf7794f, 0x987d7b, 0xf4f961, 0x1dfee1, 0x9382e9, 0xc052e4, 0xf3c5fa, 0xd6709b, 0xfe16f4, 0x34f50e, 0xab7b05, 0xfbbd13 };
 
    std::vector<uint8_t> LoadBGRAImage(const wchar_t* filename, uint32_t& width, uint32_t& height)
    {
        ComPtr<IWICImagingFactory> wicFactory;
        DX::ThrowIfFailed(CoCreateInstance(CLSID_WICImagingFactory2, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wicFactory)));

        ComPtr<IWICBitmapDecoder> decoder;
        DX::ThrowIfFailed(wicFactory->CreateDecoderFromFilename(filename, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, decoder.GetAddressOf()));

        ComPtr<IWICBitmapFrameDecode> frame;
        DX::ThrowIfFailed(decoder->GetFrame(0, frame.GetAddressOf()));

        DX::ThrowIfFailed(frame->GetSize(&width, &height));

        WICPixelFormatGUID pixelFormat;
        DX::ThrowIfFailed(frame->GetPixelFormat(&pixelFormat));

        uint32_t rowPitch = width * sizeof(uint32_t);
        uint32_t imageSize = rowPitch * height;

        std::vector<uint8_t> image;
        image.resize(size_t(imageSize));

        if (memcmp(&pixelFormat, &GUID_WICPixelFormat32bppBGRA, sizeof(GUID)) == 0)
        {
            DX::ThrowIfFailed(frame->CopyPixels(nullptr, rowPitch, imageSize, reinterpret_cast<BYTE*>(image.data())));
        }
        else
        {
            ComPtr<IWICFormatConverter> formatConverter;
            DX::ThrowIfFailed(wicFactory->CreateFormatConverter(formatConverter.GetAddressOf()));

            BOOL canConvert = FALSE;
            DX::ThrowIfFailed(formatConverter->CanConvert(pixelFormat, GUID_WICPixelFormat32bppBGRA, &canConvert));
            if (!canConvert)
            {
                throw std::exception("CanConvert");
            }

            DX::ThrowIfFailed(formatConverter->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA,
                WICBitmapDitherTypeErrorDiffusion, nullptr, 0, WICBitmapPaletteTypeMedianCut));

            DX::ThrowIfFailed(formatConverter->CopyPixels(nullptr, rowPitch, imageSize, reinterpret_cast<BYTE*>(image.data())));
        }

        return image;
    }

    // Returns true if any of the supplied floats are inf or NaN, false otherwise.
    static bool IsInfOrNan(dml::Span<const float> vals)
    {
        for (float val : vals)
        {
            if (std::isinf(val) || std::isnan(val))
            {
                return true;
            }
        }

        return false;
    }

    // Given two axis-aligned bounding boxes, computes the area of intersection divided by the area of the union of
    // the two boxes.
    static float ComputeIntersectionOverUnion(const Prediction& a, const Prediction& b)
    {
        float aArea = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        float bArea = (b.xmax - b.xmin) * (b.ymax - b.ymin);

        // Determine the bounds of the intersection rectangle
        float interXMin = std::max(a.xmin, b.xmin);
        float interYMin = std::max(a.ymin, b.ymin);
        float interXMax = std::min(a.xmax, b.xmax);
        float interYMax = std::min(a.ymax, b.ymax);

        float intersectionArea = std::max(0.0f, interXMax - interXMin) * std::max(0.0f, interYMax - interYMin);
        float unionArea = aArea + bArea - intersectionArea;

        return (intersectionArea / unionArea);
    }

    // Given a set of predictions, applies the non-maximal suppression (NMS) algorithm to select the "best" of
    // multiple overlapping predictions.
    static std::vector<Prediction> ApplyNonMaximalSuppression(dml::Span<const Prediction> allPredictions, float threshold)
    {
        std::unordered_map<uint32_t, std::vector<Prediction>> predsByClass;
        for (const auto& pred : allPredictions)
        {
            predsByClass[pred.predictedClass].push_back(pred);
        }

        std::vector<Prediction> selected;

        for (auto& kvp : predsByClass)
        {
            std::vector<Prediction>& proposals = kvp.second;

            while (!proposals.empty())
            {
                // Find the proposal with the highest score
                auto max_iter = std::max_element(proposals.begin(), proposals.end(),
                    [](const Prediction& lhs, const Prediction& rhs) {
                        return lhs.score < rhs.score;
                    });

                // Move it into the "selected" array
                selected.push_back(*max_iter);
                proposals.erase(max_iter);

                // Compare this selected prediction with all the remaining propsals. Compute their IOU and remove any
                // that are greater than the threshold.
                for (auto it = proposals.begin(); it != proposals.end(); it)
                {
                    float iou = ComputeIntersectionOverUnion(selected.back(), *it);

                    if (iou > threshold)
                    {
                        // Remove this element
                        it = proposals.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
            }
        }

        return selected;
    }

    // Helper function for fomatting strings. Format(os, a, b, c) is equivalent to os << a << b << c.
    // Helper function for fomatting strings. Format(os, aresizeer, b, c) is equivalent to os << a << b << c.
    template <typename T>
    std::ostream& Format(std::ostream& os, T&& arg)
    {
        return (os << std::forward<T>(arg));
    }

    template <typename T, typename... Ts>
    std::ostream& Format(std::ostream& os, T&& arg, Ts&&... args)
    {
        os << std::forward<T>(arg);
        return Format(os, std::forward<Ts>(args)...);
    }
    inline unsigned char* pixel(unsigned char* Img, int i, int j, int width, int height, int bpp)
    {
        return (Img + ((i * width + j) * bpp));
    }

    // Converts a pixel buffer to an NCHW tensor (batch size 1).
    // Source: buffer of RGB pixels (HWC) using uint8 components.
    // Target: buffer of RGB planes (CHW) using float32/float16 components.
    template <typename T>
    void CopyPixelsToTensor(
        byte*  src,
        uint32_t srcWidth, uint32_t srcHeight, uint32_t rowPitch,
        dml::Span<std::byte> dst,
        uint32_t height,
        uint32_t width,
        uint32_t channels)
    {
        uint32_t srcChannels = rowPitch / srcWidth;
        uint32_t rowWidth = rowPitch / srcChannels;
        dml::Span<T> dstT(reinterpret_cast<T*>(dst.data()), dst.size_bytes() / sizeof(T));

        if (srcWidth != width || srcHeight != height)
        {
            unsigned char* Img = src;
            float ScaledWidthRatio = srcWidth / (float)width;
            float ScaledHeightRatio = srcHeight / (float)height;
            uint32_t pixelIndex = 0;

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    float mappedheight = i * ScaledHeightRatio;  //rf
                    float mappedwidth = j * ScaledWidthRatio;   //cf
                    int   OriginalPosHeight = (int)mappedheight;         //ro
                    int   OriginalPosWidth =(int) mappedwidth;          //co
                    float deltaheight = mappedheight - OriginalPosHeight; //delta r
                    float deltawidth = mappedwidth - OriginalPosWidth;   //delta c

                    unsigned char* temp1 = pixel(Img, OriginalPosHeight, OriginalPosWidth, rowWidth, srcHeight, srcChannels);
                    unsigned char* temp2 = pixel(Img, ((OriginalPosHeight + 1) >= height ? OriginalPosHeight : OriginalPosHeight + 1),
                        OriginalPosWidth, rowWidth, srcHeight, srcChannels);
                    unsigned char* temp3 = pixel(Img, OriginalPosHeight, OriginalPosWidth + 1, rowWidth, srcHeight, srcChannels);
                    unsigned char* temp4 = pixel(Img, ((OriginalPosHeight + 1) >= srcHeight ? OriginalPosHeight : OriginalPosHeight + 1),
                        (OriginalPosWidth + 1) >= rowWidth ? OriginalPosWidth : (OriginalPosWidth + 1), rowWidth, srcHeight, srcChannels);

                    float b =
                        (*(temp1 + 0) * (1 - deltaheight) * (1 - deltawidth) \
                        + *(temp2 + 0) * (deltaheight) * (1 - deltawidth) \
                        + *(temp3 + 0) * (1 - deltaheight) * (deltawidth) \
                        + *(temp4 + 0) * (deltaheight) * (deltawidth)) / 255.0f;

                    float g =
                        (*(temp1 + 1) * (1 - deltaheight) * (1 - deltawidth) \
                        + *(temp2 + 1) * (deltaheight) * (1 - deltawidth) \
                        + *(temp3 + 1) * (1 - deltaheight) * (deltawidth) \
                        + *(temp4 + 1) * (deltaheight) * (deltawidth)) / 255.0f;

                    float r =
                        (*(temp1 + 2) * (1 - deltaheight) * (1 - deltawidth) \
                        + *(temp2 + 2) * (deltaheight) * (1 - deltawidth) \
                        + *(temp3 + 2) * (1 - deltaheight) * (deltawidth) \
                        + *(temp4 + 2) * (deltaheight) * (deltawidth)) / 255.0f;


                    dstT[pixelIndex + 0 * height * width] = r;
                    dstT[pixelIndex + 1 * height * width] = g;
                    dstT[pixelIndex + 2 * height * width] = b;
                    pixelIndex++;
                }
            }
        }
        else
        {
            double rs, gs, bs;
            rs = gs = bs = 0.0;
            for (size_t pixelIndex = 0; pixelIndex < height * width; pixelIndex++)
            {
                float b = static_cast<float>(src[pixelIndex * srcChannels + 0]) / 255.0f;
                float g = static_cast<float>(src[pixelIndex * srcChannels + 1]) / 255.0f;
                float r = static_cast<float>(src[pixelIndex * srcChannels + 2]) / 255.0f;

                //rs += r;
                //gs += g;
                //bs += b;

                
                dstT[pixelIndex + 0 * height * width] = r;
                dstT[pixelIndex + 1 * height * width] = g;
                dstT[pixelIndex + 2 * height * width] = b;
            }

            //rs /= height * width;
            //gs /= height * width;
            //bs /= height * width;
            //std::stringstream ss;
            //Format(ss, "red: ", rs, ", green: ", gs, ", blue: ", bs,  "\n");
            //OutputDebugStringA(ss.str().c_str());
        }
    }

    static void WaitForQueueToComplete(ID3D12CommandQueue* queue) {
        ComPtr<ID3D12Device> device;
        THROW_IF_FAILED(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
        ComPtr<ID3D12Fence> fence;
        THROW_IF_FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
        THROW_IF_FAILED(queue->Signal(fence.Get(), 1));
        THROW_IF_FAILED(fence->SetEventOnCompletion(1, nullptr));
    }
    static ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
        ID3D12Device* d3d_device,
        size_t resource_byte_size,
        D3D12_HEAP_TYPE heap_type = D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATES resource_state = D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAGS resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
        resource_byte_size = std::max(resource_byte_size, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

        // DML needs the resources' sizes to be a multiple of 4 bytes
        (resource_byte_size += 3) &= ~3;

        D3D12_HEAP_PROPERTIES heap_properties = {};
        heap_properties.Type = heap_type;
        heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap_properties.CreationNodeMask = 1;
        heap_properties.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC resource_desc = {};
        resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource_desc.Alignment = 0;
        resource_desc.Width = static_cast<uint64_t>(resource_byte_size);
        resource_desc.Height = 1;
        resource_desc.DepthOrArraySize = 1;
        resource_desc.MipLevels = 1;
        resource_desc.Format = DXGI_FORMAT_UNKNOWN;
        resource_desc.SampleDesc = { 1, 0 };
        resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resource_desc.Flags = resource_flags;

        ComPtr<ID3D12Resource> gpu_resource;
        THROW_IF_FAILED(d3d_device->CreateCommittedResource(
            &heap_properties,
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            resource_state,
            nullptr,
            IID_PPV_ARGS(&gpu_resource)));

        return gpu_resource;
    }



    enum class ChannelOrder
    {
        RGB,
        BGR,
    };

   

}


bool Sample::CopySharedVideoTextureTensor(std::vector<std::byte> & inputBuffer)
{
    // Record start
    auto start = std::chrono::high_resolution_clock::now();

    auto hr = m_d3dDevice->GetDeviceRemovedReason();
    ComPtr<ID3D11Texture2D> mediaTexture;
    if (SUCCEEDED(m_player->GetDevice()->OpenSharedResource1(m_sharedVideoTexture, IID_PPV_ARGS(mediaTexture.GetAddressOf()))))
    {
        // First verify that we can map the texture
        D3D11_TEXTURE2D_DESC desc;
        mediaTexture->GetDesc(&desc);

        // translate texture format to WIC format. We support only BGRA and ARGB.
        GUID wicFormatGuid;
        switch (desc.Format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            wicFormatGuid = GUID_WICPixelFormat32bppRGBA;
            break;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            wicFormatGuid = GUID_WICPixelFormat32bppBGRA;
            break;
        default:
            throw std::exception("Unsupported DXGI_FORMAT: %d. Only RGBA and BGRA are supported.");
        }

        // Get the device context
        ComPtr<ID3D11Device> d3dDevice;
        mediaTexture->GetDevice(&d3dDevice);

        auto hr = d3dDevice->GetDeviceRemovedReason();
        ComPtr<ID3D11DeviceContext> d3dContext;
        d3dDevice->GetImmediateContext(&d3dContext);

        // map the texture
        ComPtr<ID3D11Texture2D> mappedTexture;
        D3D11_MAPPED_SUBRESOURCE mapInfo;
        mapInfo.RowPitch;
        hr = d3dContext->Map(
            mediaTexture.Get(),
            0,  // Subresource
            D3D11_MAP_READ,
            0,  // MapFlags
            &mapInfo);

        if (FAILED(hr)) {
            // If we failed to map the texture, copy it to a staging resource
            if (hr == E_INVALIDARG) {
                D3D11_TEXTURE2D_DESC desc2;
                desc2.Width = desc.Width;
                desc2.Height = desc.Height;
                desc2.MipLevels = desc.MipLevels;
                desc2.ArraySize = desc.ArraySize;
                desc2.Format = desc.Format;
                desc2.SampleDesc = desc.SampleDesc;
                desc2.Usage = D3D11_USAGE_STAGING;
                desc2.BindFlags = 0;
                desc2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                desc2.MiscFlags = 0;

                ComPtr<ID3D11Texture2D> stagingTexture;
                DX::ThrowIfFailed(m_player->GetDevice()->CreateTexture2D(&desc2, nullptr, &stagingTexture));

                // copy the texture to a staging resource
                d3dContext->CopyResource(stagingTexture.Get(), mediaTexture.Get());

                // now, map the staging resource
                hr = d3dContext->Map(
                    stagingTexture.Get(),
                    0,
                    D3D11_MAP_READ,
                    0,
                    &mapInfo);
                if (FAILED(hr)) {
                    throw  std::exception("Failed to map staging texture");
                }

                mappedTexture = std::move(stagingTexture);
            }
            else {
                throw std::exception("Failed to map texture.");
            }
        }
        else {
            mappedTexture = mediaTexture;
        }
        auto unmapResource = Finally([&] {
            d3dContext->Unmap(mappedTexture.Get(), 0);
            });

       
     
        const size_t inputChannels = m_inputShape[m_inputShape.size() - 3];
        const size_t inputHeight = m_inputShape[m_inputShape.size() - 2];
        const size_t inputWidth = m_inputShape[m_inputShape.size() - 1];

        if (desc.Width != inputWidth || desc.Height != inputHeight)
        {
            if (m_d2d1_factory.Get() == nullptr)
            {
                // Create a Direct2D factory.
                HRESULT hr = D2D1CreateFactory(
                    D2D1_FACTORY_TYPE_MULTI_THREADED,
                    m_d2d1_factory.GetAddressOf());

                // Create D2Device 
                auto device = m_deviceResources->GetD3DDevice();
                ComPtr<IDXGIDevice3> dxgiDevice;
                DX::ThrowIfFailed(
                    d3dDevice.As(&dxgiDevice)
                );

                m_d2d1_factory->CreateDevice(
                    dxgiDevice.Get(),
                    m_d2d1_device.GetAddressOf()
                );

                // Get Direct2D device's corresponding device context object.
                DX::ThrowIfFailed(
                    m_d2d1_device->CreateDeviceContext(
                        D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
                        &m_d2dContext
                    )
                );


            }
            if (m_d2dContext.Get())
            {
                ComPtr < IDXGISurface> surface;
               
                DX::ThrowIfFailed(
                    mediaTexture.As(&surface)
                );
               
                ComPtr< ID2D1Bitmap1> bitmap;
                m_d2dContext.Get()->CreateBitmapFromDxgiSurface(surface.Get(), NULL, bitmap.GetAddressOf());


                ComPtr<ID2D1Effect> scaleEffect;
                m_d2dContext->CreateEffect(CLSID_D2D1Scale, &scaleEffect);

                scaleEffect->SetInput(0, bitmap.Get());

               
                D2D1_SCALE_INTERPOLATION_MODE interpolationMode = D2D1_SCALE_INTERPOLATION_MODE_HIGH_QUALITY_CUBIC;
                scaleEffect->SetValue(D2D1_SCALE_PROP_SCALE, D2D1::Vector2F((float)inputWidth/(float)desc.Width, (float)inputHeight / (float)desc.Height));
                scaleEffect->SetValue(D2D1_SCALE_PROP_INTERPOLATION_MODE, reinterpret_cast<const BYTE*>(&interpolationMode), sizeof(D2D1_SCALE_INTERPOLATION_MODE)); // Set the interpolation mode.

                ComPtr< ID2D1Image> image_out;
                scaleEffect->GetOutput(image_out.GetAddressOf());

                D3D11_TEXTURE2D_DESC texDesc;
                texDesc.Width = static_cast<UINT>(inputWidth);
                texDesc.Height = static_cast<UINT>(inputHeight);
                texDesc.MipLevels = 1;
                texDesc.ArraySize = 1;
                texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                texDesc.SampleDesc.Count = 1;
                texDesc.SampleDesc.Quality = 0;
                texDesc.Usage = D3D11_USAGE_DEFAULT;
                texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                texDesc.CPUAccessFlags = 0;
                texDesc.MiscFlags = 0;

                ComPtr<ID3D11Texture2D> drawTexture;
                DX::ThrowIfFailed(m_player->GetDevice()->CreateTexture2D(&texDesc, nullptr, &drawTexture));

                ComPtr < IDXGISurface> drawSurface;
                DX::ThrowIfFailed(
                    drawTexture.As(&drawSurface)
                );
               
                ComPtr< ID2D1Bitmap1> drawBitmap;
                m_d2dContext.Get()->CreateBitmapFromDxgiSurface(drawSurface.Get(), NULL, drawBitmap.GetAddressOf());
                m_d2dContext.Get()->SetTarget(drawBitmap.Get());

                // Draw the image into the device context. Output surface is set as the target of the device context.
                m_d2dContext.Get()->BeginDraw();
               
                auto identityMat = D2D1::Matrix3x2F::Identity();

                m_d2dContext.Get()->SetTransform(identityMat);     // Clear out any existing transform before drawing.
                D2D1_POINT_2F targetOffset = { 0, 0 };
                m_d2dContext.Get()->DrawImage(image_out.Get(), targetOffset);
                D2D1_TAG tag1;
                D2D1_TAG tag2;
                auto hr = m_d2dContext.Get()->EndDraw(&tag1, &tag2);
                m_d2dContext.Get()->SetTarget(nullptr);
                 

                // we have our scaled image in drawTexture

                // create staging texture
                D3D11_TEXTURE2D_DESC desc2;
                desc2.Width = texDesc.Width;
                desc2.Height = texDesc.Height;
                desc2.MipLevels = texDesc.MipLevels;
                desc2.ArraySize = texDesc.ArraySize;
                desc2.Format = texDesc.Format;
                desc2.SampleDesc = texDesc.SampleDesc;
                desc2.Usage = D3D11_USAGE_STAGING;
                desc2.BindFlags = 0;
                desc2.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                desc2.MiscFlags = 0;

                ComPtr<ID3D11Texture2D> stagingTexture;
                DX::ThrowIfFailed(m_player->GetDevice()->CreateTexture2D(&desc2, nullptr, &stagingTexture));

                // copy the texture to a staging resource
                d3dContext->CopyResource(stagingTexture.Get(), drawTexture.Get());

                // now, map the staging resource
                hr = d3dContext->Map(
                    stagingTexture.Get(),
                    0,
                    D3D11_MAP_READ,
                    0,
                    &mapInfo);
                if (FAILED(hr)) {
                    throw  std::exception("Failed to map staging texture");
                }

                mappedTexture = std::move(stagingTexture);
                mappedTexture->GetDesc(&desc);
              
            }

        }

       
        switch (m_inputDataType)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            CopyPixelsToTensor<float>((byte*)mapInfo.pData, desc.Width, desc.Height, mapInfo.RowPitch, inputBuffer, inputHeight, inputWidth, inputChannels);
            break;

        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            CopyPixelsToTensor<half_float::half>((byte*)mapInfo.pData, desc.Width, desc.Height, mapInfo.RowPitch, inputBuffer, inputHeight, inputWidth, inputChannels);
            break;

        default:
            throw std::invalid_argument("Unsupported data type");
        }   
        auto end = std::chrono::high_resolution_clock::now();
        m_copypixels_tensor_duration = end - start;
        return true;
    }
    return false;
}

void Sample::GetFaces(std::vector<const std::byte*>& outputData, std::vector<std::vector<int64_t>>& shapes)
{
    if (outputData.size() != 2)
        return;

    Vec3<float> value1((float*)outputData[0], shapes[0][0], shapes[0][1], shapes[0][2]);
    Vec3<float> value2((float*)outputData[1], shapes[1][0], shapes[1][1], shapes[1][2]);
 
    // Scale the boxes to be relative to the original image size
    auto viewport = m_deviceResources->GetScreenViewport();
    float xScale = (float)viewport.Width;
    float yScale = (float)viewport.Height;

    const float x_scale = (float)m_inputWidth;
    const float y_scale = (float)m_inputHeight;
    const float h_scale = (float)m_inputWidth;
    const float w_scale = (float)m_inputHeight;

    for (Size i = 0; i < value1.z; ++i)
    {
        for (Size j = 0; j < value1.y; ++j)
        {
            auto ptr2 = value2[i][j];
            if (ptr2[0] < threshold) continue;

            auto ptr = value1[i][j];

            Detection result;
            result.x = ptr[0];
            result.y = ptr[1];
            result.w = ptr[2];
            result.h = ptr[3];
            result.index = -1; // face, no label
            result.confidence = (float)ptr2[0];

            result.x = result.x / x_scale * m_anchors[j].w + m_anchors[j].x_center;
            result.y = result.y / y_scale * m_anchors[j].h + m_anchors[j].y_center;

            result.h = result.h / h_scale * m_anchors[j].h;
            result.w = result.w / w_scale * m_anchors[j].w;

            // We need to do some postprocessing on the raw values before we return them

            // Convert x,y,w,h to xmin,ymin,xmax,ymax
            
            float xmin = result.x - result.w / 2;
            float ymin = result.y - result.h / 2;
            float xmax = result.x + result.w / 2;
            float ymax = result.y + result.h / 2;

            xmin *= xScale;
            ymin *= yScale;
            xmax *= xScale;
            ymax *= yScale;

            // Clip values out of range
            xmin = std::clamp(xmin, 0.0f, (float)viewport.Width);
            ymin = std::clamp(ymin, 0.0f, (float)viewport.Height);
            xmax = std::clamp(xmax, 0.0f, (float)viewport.Width);
            ymax = std::clamp(ymax, 0.0f, (float)viewport.Height);

            // Discard invalid boxes
            if (xmax <= xmin || ymax <= ymin || IsInfOrNan({ xmin, ymin, xmax, ymax }))
            {
                continue;
            }

            Prediction pred = {};
            pred.xmin = xmin;
            pred.ymin = ymin;
            pred.xmax = xmax;
            pred.ymax = ymax;
            pred.score = result.confidence;
            pred.predictedClass = result.index;

            for (int i = 0; i < 6; i++)
            {
                float keypoint_x = ptr[4 + i * 2];
                float keypoint_y = ptr[5 + i * 2];

                float x = keypoint_x / x_scale * m_anchors[j].w + m_anchors[j].x_center;
                float y = keypoint_y / y_scale * m_anchors[j].h + m_anchors[j].y_center;
                x *= xScale;
                y *= yScale;
                pred.m_keypoints.push_back(std::pair<float, float>(x, y));
            }

            m_preds.emplace_back(pred);
        }
    }
    // Apply NMS to select the best boxes
    m_preds = ApplyNonMaximalSuppression(m_preds, YoloV4Constants::c_nmsThreshold);
}


void Sample::GetPredictions(std::vector<const std::byte*>& outputData, std::vector<std::vector<int64_t>>& shapes)
{
    if (outputData.size() != 3)
        return;

    Vec3<float> value1((float*)outputData[0], shapes[0][0], shapes[0][1], shapes[0][2]);
    Vec2<float> value2((float*)outputData[1], shapes[1][0], shapes[1][1]);
    Vec2<float> value3((float*)outputData[2], shapes[2][0], shapes[2][1]);

    // Scale the boxes to be relative to the original image size
    auto viewport = m_deviceResources->GetScreenViewport();
    float xScale = (float)viewport.Width / m_inputWidth;
    float yScale = (float)viewport.Height / m_inputHeight;

    for (Size i = 0; i < value1.z; ++i)
    {
        for (Size j = 0; j < value1.y; ++j)
        {
            auto ptr2 = value2[i][j];
            if (ptr2 < threshold) continue;

            auto ptr = value1[i][j];
           
            Detection result;
            result.x = ptr[0];
            result.y = ptr[1];
            result.w = ptr[2];
            result.h = ptr[3];
            result.index = (Size)value3[i][j];
            result.confidence = (float)ptr2;

            // We need to do some postprocessing on the raw values before we return them

            // Convert x,y,w,h to xmin,ymin,xmax,ymax
            float xmin = result.x;
            float ymin = result.y;
            float xmax = result.w;
            float ymax = result.h;

            xmin *= xScale;
            ymin *= yScale;
            xmax *= xScale;
            ymax *= yScale;

            // Clip values out of range
            xmin = std::clamp(xmin, 0.0f, (float)viewport.Width);
            ymin = std::clamp(ymin, 0.0f, (float)viewport.Height);
            xmax = std::clamp(xmax, 0.0f, (float)viewport.Width);
            ymax = std::clamp(ymax, 0.0f, (float)viewport.Height);

            // Discard invalid boxes
            if (xmax <= xmin || ymax <= ymin || IsInfOrNan({ xmin, ymin, xmax, ymax }))
            {
                continue;
            }

            Prediction pred = {};
            pred.xmin = xmin;
            pred.ymin = ymin;
            pred.xmax = xmax;
            pred.ymax = ymax;
            pred.score = result.confidence;
            pred.predictedClass = result.index;
            m_preds.push_back(pred);
        }
    }
    // Apply NMS to select the best boxes
    m_preds = ApplyNonMaximalSuppression(m_preds, YoloV4Constants::c_nmsThreshold);
}


void Sample::GetPredictions(const std::byte *  outputData, std::vector<int64_t> & shape)
{
    Vec3<float> value((float*)outputData, shape[0], shape[1], shape[2]);

    // Scale the boxes to be relative to the original image size
    auto viewport = m_deviceResources->GetScreenViewport();
    float xScale = (float)viewport.Width / YoloV4Constants::c_inputWidth;
    float yScale = (float)viewport.Height / YoloV4Constants::c_inputHeight;

    for (Size i = 0; i < value.z; ++i)
    {
        for (Size j = 0; j < value.y; ++j)
        {
            auto ptr = value[i][j];
            if (ptr[4] < threshold) continue;
            Detection result;
            result.x = ptr[0];
            result.y = ptr[1];
            result.w = ptr[2];
            result.h = ptr[3];
            result.index = (Size)ptr[5];
            result.confidence = (float)ptr[4];

            // We need to do some postprocessing on the raw values before we return them

            // Convert x,y,w,h to xmin,ymin,xmax,ymax
            float xmin = result.x - result.w / 2;
            float ymin = result.y - result.h / 2;
            float xmax = result.x + result.w / 2;
            float ymax = result.y + result.h / 2;

            xmin *= xScale;
            ymin *= yScale;
            xmax *= xScale;
            ymax *= yScale;

            // Clip values out of range
            xmin = std::clamp(xmin, 0.0f, (float)viewport.Width);
            ymin = std::clamp(ymin, 0.0f, (float)viewport.Height);
            xmax = std::clamp(xmax, 0.0f, (float)viewport.Width);
            ymax = std::clamp(ymax, 0.0f, (float)viewport.Height);

            // Discard invalid boxes
            if (xmax <= xmin || ymax <= ymin || IsInfOrNan({ xmin, ymin, xmax, ymax }))
            {
                continue;
            }

            Prediction pred = {};
            pred.xmin = xmin;
            pred.ymin = ymin;
            pred.xmax = xmax;
            pred.ymax = ymax;
            pred.score = result.confidence;
            pred.predictedClass = result.index;
            m_preds.push_back(pred);
        }
    }
    // Apply NMS to select the best boxes
    m_preds = ApplyNonMaximalSuppression(m_preds, YoloV4Constants::c_nmsThreshold);
}

Sample::Sample()
    : m_ctrlConnected(false)
{
    // Use gamma-correct rendering.
    // Renders only 2D, so no need for a depth buffer.
    m_deviceResources = std::make_unique<DX::DeviceResources>(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
        3, D3D_FEATURE_LEVEL_11_0, DX::DeviceResources::c_AllowTearing);
    m_deviceResources->RegisterDeviceNotify(this);
}

Sample::~Sample()
{
    if (m_deviceResources)
    {
        m_deviceResources->WaitForGpu();
    }
}

// Initialize the Direct3D resources required to run.
bool Sample::Initialize(HWND window, int width, int height)
{
    m_gamePad = std::make_unique<GamePad>();

    m_keyboard = std::make_unique<Keyboard>();
    
    m_deviceResources->SetWindow(window, width, height);

    InitializeDirectML(
        m_d3dDevice.GetAddressOf(),
        m_commandQueue.GetAddressOf(),
        m_dmlDevice.GetAddressOf(),
        m_commandAllocator.GetAddressOf(), 
        m_commandList.GetAddressOf());

    // Add the DML execution provider to ORT using the DML Device and D3D12 Command Queue created above.
    if (!m_dmlDevice)
    {
        MessageBox(0, L"No NPU device found\n", L"Error", MB_OK);
        ExitProcess(1);
        return false;
    }


    InitializeDirectMLResources();

    m_deviceResources->CreateDeviceResources();  	
    CreateDeviceDependentResources();
   
    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();

    return true;
}

#pragma region Frame Update
// Executes basic render loop.
void Sample::Tick()
{
    m_timer.Tick([&]()
    {
        Update(m_timer);
    });

    Render();
}

// Updates the world.
void Sample::Update(DX::StepTimer const& timer)
{
    PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

    float elapsedTime = float(timer.GetElapsedSeconds());

    m_fps.Tick(elapsedTime);

    auto pad = m_gamePad->GetState(0);
    if (pad.IsConnected())
    {
        m_ctrlConnected = true;

        m_gamePadButtons.Update(pad);

        if (pad.IsViewPressed())
        {
            ExitSample();
        }

        if (m_gamePadButtons.x == DirectX::GamePad::ButtonStateTracker::PRESSED && m_player.get() != nullptr)
        {
            if (m_player->IsPlaying())
            {
                m_player->Pause();
            }
            else
            {
                m_player->Play();
            }
        }
    }
    else
    {
        m_ctrlConnected = false;
        m_gamePadButtons.Reset();
    }

    auto kb = m_keyboard->GetState();
    m_keyboardButtons.Update(kb);

    if (kb.Escape)
    {
        ExitSample();
    }

    if (m_keyboardButtons.IsKeyPressed(Keyboard::Enter) && m_player.get() != nullptr)
    {
        if (m_player->IsPlaying())
        {
            m_player->Pause();
        }
        else
        {
            m_player->Play();
        }
    }
    if (m_keyboardButtons.IsKeyPressed(Keyboard::Right) && m_player.get() != nullptr)
    {
        m_player->Pause();
        if (m_keyboardButtons.IsKeyPressed(Keyboard::LeftShift) || m_keyboardButtons.IsKeyPressed(Keyboard::RightShift))
            m_player->Skip(30);
        else
            m_player->Skip(10);
        m_player->Play();
    }
    if (m_keyboardButtons.IsKeyPressed(Keyboard::Left) && m_player.get() != nullptr)
    {
        m_player->Pause();
        if (m_keyboardButtons.IsKeyPressed(Keyboard::LeftShift) || m_keyboardButtons.IsKeyPressed(Keyboard::RightShift))
            m_player->Skip(-30);
        else
            m_player->Skip(-10);
        m_player->Play();
    }

    PIXEndEvent();
}
#pragma endregion

void Sample::OnNewMopdel(wchar_t* modelfile)
{

    if (m_player->IsPlaying())
    {
        m_player->Pause();
    }

    InitializeDirectMLResources(modelfile);

    while (!m_player->IsInfoReady())
    {
        SwitchToThread();
    }

    m_player->Play();
    m_player->Skip(-5);
}


void Sample::OnNewFile(wchar_t* filename)
{
    if (m_player->IsPlaying())
    {
        m_player->Pause();
    }

    m_player->SetSource(filename);

    while (!m_player->IsInfoReady())
    {
        SwitchToThread();
    }

    m_player->GetNativeVideoSize(m_origTextureWidth, m_origTextureHeight);
    m_player->SetLoop(true);

    // Create texture to receive video frames.
    CD3DX12_RESOURCE_DESC desc(
        D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        0,
        m_origTextureWidth,
        m_origTextureHeight,
        1,
        1,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        1,
        0,
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);

    CD3DX12_HEAP_PROPERTIES defaultHeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto device = m_deviceResources->GetD3DDevice();

    DX::ThrowIfFailed(
        device->CreateCommittedResource(
            &defaultHeapProperties,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS(m_videoTexture.ReleaseAndGetAddressOf())));

    DX::ThrowIfFailed(
        device->CreateSharedHandle(
            m_videoTexture.Get(),
            nullptr,
            GENERIC_ALL,
            nullptr,
            &m_sharedVideoTexture));

    CreateShaderResourceView(device, m_videoTexture.Get(), m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));


    m_player->Play();
}


#pragma region Frame Render
// Draws the scene.
void Sample::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

    // Prepare the command list to render a new frame.
    m_deviceResources->Prepare();
    Clear();

    auto commandList = m_deviceResources->GetCommandList();

    // Render the result to the screen

    auto viewport = m_deviceResources->GetScreenViewport();
    auto scissorRect = m_deviceResources->GetScissorRect();

    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render to screen");

        commandList->OMSetRenderTargets(1, &m_deviceResources->GetRenderTargetView(), FALSE, nullptr);

        commandList->SetGraphicsRootSignature(m_texRootSignatureLinear.Get());
        commandList->SetPipelineState(m_texPipelineStateLinear.Get());

        auto heap = m_SRVDescriptorHeap->Heap();
        commandList->SetDescriptorHeaps(1, &heap);

        commandList->SetGraphicsRootDescriptorTable(0,
            m_SRVDescriptorHeap->GetGpuHandle(e_descTexture));

        // Set necessary state.
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->IASetIndexBuffer(&m_indexBufferView);

        // Draw full screen texture
        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);
        commandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);

        commandList->DrawIndexedInstanced(6, 1, 0, 0, 0);

        PIXEndEvent(commandList);
    }

    // Render the UI
    {
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render UI");

        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);

        auto size = m_deviceResources->GetOutputSize();
        auto safe = SimpleMath::Viewport::ComputeTitleSafeArea(size.right, size.bottom);

        // Draw the text HUD.
        ID3D12DescriptorHeap* fontHeaps[] = { m_fontDescriptorHeap->Heap() };
        commandList->SetDescriptorHeaps(_countof(fontHeaps), fontHeaps);

        m_spriteBatch->Begin(commandList);

        float xCenter = static_cast<float>(safe.left + (safe.right - safe.left) / 2);

        const wchar_t* mainLegend = m_ctrlConnected ?
            L"[View] Exit   [X] Play/Pause"
            : L"ESC - Exit     ENTER - Play/Pause   Mouse Context Menu - Open new Video (click above) / Onnx-Model (click on this line)   (Shift) < or (Shift) > - back- forward";
        SimpleMath::Vector2 mainLegendSize = m_legendFont->MeasureString(mainLegend);
        auto mainLegendPos = SimpleMath::Vector2(xCenter - mainLegendSize.x / 2, static_cast<float>(safe.bottom) - m_legendFont->GetLineSpacing());

        // Render a drop shadow by drawing the text twice with a slight offset.
        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
            mainLegend, mainLegendPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.0f, 0.0f, 0.0f, 0.25f));
        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
            mainLegend, mainLegendPos, ATG::Colors::White);

        const wchar_t* modeLabel = L"Object detection model:";
        SimpleMath::Vector2 modeLabelSize = m_labelFontBold->MeasureString(modeLabel);
        auto modeLabelPos = SimpleMath::Vector2(safe.right - modeLabelSize.x, static_cast<float>(safe.top));

        m_labelFontBold->DrawString(m_spriteBatch.get(), modeLabel, modeLabelPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFontBold->DrawString(m_spriteBatch.get(), modeLabel, modeLabelPos, ATG::Colors::White);

        wchar_t model[128];
        swprintf_s(model, 128, L"%s NPU", m_modelfile.c_str());
        SimpleMath::Vector2 modelSize = m_labelFont->MeasureString(model);
        auto modelPos = SimpleMath::Vector2(safe.right - modelSize.x, static_cast<float>(safe.top) + m_labelFontBold->GetLineSpacing());

        m_labelFont->DrawString(m_spriteBatch.get(), model, modelPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), model, modelPos, ATG::Colors::White);

        wchar_t fps[16];
        swprintf_s(fps, 16, L"%0.2f FPS", m_fps.GetFPS());
        SimpleMath::Vector2 fpsSize = m_labelFont->MeasureString(fps);
        auto fpsPos = SimpleMath::Vector2(safe.right - fpsSize.x, static_cast<float>(safe.top) + m_labelFont->GetLineSpacing() * 3.f);

        m_labelFont->DrawString(m_spriteBatch.get(), fps, fpsPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), fps, fpsPos, ATG::Colors::White);

        wchar_t cpt[32];
        swprintf_s(cpt, 32, L"scale/copy: %0.2f ms", m_copypixels_tensor_duration.count());
        SimpleMath::Vector2 cptySize = m_labelFont->MeasureString(cpt);
        auto cptyPos = SimpleMath::Vector2(safe.right - cptySize.x, static_cast<float>(safe.top) + m_labelFont->GetLineSpacing() * 4.f);

        m_labelFont->DrawString(m_spriteBatch.get(), cpt, cptyPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), cpt, cptyPos, ATG::Colors::White);

        wchar_t inf[32];
        swprintf_s(inf, 32, L"inference: %0.2f ms", m_inference_duration.count());
        SimpleMath::Vector2 infySize = m_labelFont->MeasureString(inf);
        auto infyPos = SimpleMath::Vector2(safe.right - infySize.x, static_cast<float>(safe.top) + m_labelFont->GetLineSpacing() * 5.f);

        m_labelFont->DrawString(m_spriteBatch.get(), inf, infyPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), inf, infyPos, ATG::Colors::White);

        wchar_t out[32];
        swprintf_s(out, 32, L"output: %0.2f ms", m_output_duration.count());
        SimpleMath::Vector2 outySize = m_labelFont->MeasureString(out);
        auto outyPos = SimpleMath::Vector2(safe.right - outySize.x, static_cast<float>(safe.top) + m_labelFont->GetLineSpacing() * 6.f);

        m_labelFont->DrawString(m_spriteBatch.get(), out, outyPos + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.f, 0.f, 0.f, 0.25f));
        m_labelFont->DrawString(m_spriteBatch.get(), out, outyPos, ATG::Colors::White);

        m_spriteBatch->End();

        PIXEndEvent(commandList);
    }
    // 
    // Kick off the compute work that will be used to render the next frame. We do this now so that the data will be
    // ready by the time the next frame comes around.
    // 

#if USE_VIDEO
    // Get the latest video frame
    RECT r = { 0, 0, static_cast<LONG>(m_origTextureWidth), static_cast<LONG>(m_origTextureHeight) };
    MFVideoNormalizedRect rect = { 0.0f, 0.0f, 1.0f, 1.0f };
    m_player->TransferFrame(m_sharedVideoTexture, rect, r);
#endif

    // Convert image to tensor format (original texture -> model input)
    const size_t inputChannels = m_inputShape[m_inputShape.size() - 3];
    const size_t inputHeight = m_inputShape[m_inputShape.size() - 2];
    const size_t inputWidth = m_inputShape[m_inputShape.size() - 1];
    const size_t inputElementSize = m_inputDataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float) : sizeof(uint16_t);

    // next version
    // // Create d3d_buffer using D3D12 APIs
    //    Microsoft::WRL::ComPtr<ID3D12Resource> d3d_buffer = ...;
    // 
    //  use hlsl ImageToTensor.hlsl compute shader to copy textture to input format texture d3d_buffer
    // 
    // // Create the dml resource from the D3D resource.
    //    ort_dml_api->CreateGPUAllocationFromD3DResource(d3d_buffer.Get(), &dml_resource);
    // 
    // Ort::Value ort_value(Ort::Value::CreateTensor(memory_info_dml, dml_resource,
    //         d3d_buffer_size, shape.data(), shape.size(),
    //            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    // 
    // see: https://github.com/ankan-ban/HelloOrtDml/blob/main/Main.cpp
    // https://onnxruntime.ai/docs/performance/device-tensor.html
    // Load image and transform it into an NCHW tensor with the correct shape and data type.
    std::vector<std::byte> inputBuffer(inputChannels* inputHeight* inputWidth* inputElementSize);
   
    if (CopySharedVideoTextureTensor(inputBuffer))
    {

        // Record start
        auto start = std::chrono::high_resolution_clock::now();

        // Create input tensor
        Ort::MemoryInfo memoryInfo2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        //Ort::MemoryInfo memoryInfo2("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor(
            memoryInfo2,
            inputBuffer.data(),
            inputBuffer.size(),
            m_inputShape.data(),
            m_inputShape.size(),
            m_inputDataType
        );

       

        // Bind tensors
        Ort::MemoryInfo memoryInfo0 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Allocator allocator0(m_session, memoryInfo0);
        auto inputName = m_session.GetInputNameAllocated(0, allocator0);
        auto bindings = Ort::IoBinding::IoBinding(m_session);
        try {
           
            bindings.BindInput(inputName.get(), inputTensor);
        }
        catch (const std::runtime_error& re) {
            const char* err = re.what();
            MessageBoxA(0, err, "Error loading model", MB_YESNO);
            std::cerr << "Runtime error: " << re.what() << std::endl;
            exit(1);
        }
        // Create output tensor(s) and bind
        auto tensors = m_session.GetOutputCount();
        std::vector<std::string> output_names;
        std::vector<std::vector<int64_t>> output_shapes;
        for (int i = 0; i < tensors; i++)
        {
            auto output_name = m_session.GetOutputNameAllocated(i, allocator0);
            output_names.push_back(output_name.get());
            auto type_info = m_session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            output_shapes.push_back(shape);
            bindings.BindOutput(output_names.back().c_str(), memoryInfo2);
        }
        HRESULT hr0;
        try {
            // Record start
            //auto start = std::chrono::high_   resolution_clock::now();

            // Run the session to get inference results.
            Ort::RunOptions runOpts;
            m_session.Run(runOpts, bindings);
            
            hr0 = m_d3dDevice->GetDeviceRemovedReason();

            bindings.SynchronizeOutputs();
        }
        catch (const std::runtime_error& re) {
            const char* err = re.what();
            MessageBoxA(0, err, "Error loading model", MB_YESNO);
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


        try {

            THROW_IF_FAILED(m_d3dDevice->GetDeviceRemovedReason());
        }
        catch (const std::exception & ex)
        {
            const char* err = ex.what();
            MessageBoxA(0, err, "Error loading model", MB_YESNO);
            std::cerr << "Error occurred: " << ex.what() << std::endl;
            exit(1);
            //extern void MyDeviceRemovedHandler(ID3D12Device * pDevice);
            //MyDeviceRemovedHandler(m_d3dDevice.Get());
        }
        // Queue fence, and wait for completion

        ComPtr<ID3D12Fence> fence;
        THROW_IF_FAILED(m_d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
        THROW_IF_FAILED(m_commandQueue->Signal(fence.Get(), 1));

        wil::unique_handle fenceEvent(CreateEvent(nullptr, FALSE, FALSE, nullptr));
        THROW_IF_FAILED(fence->SetEventOnCompletion(1, fenceEvent.get()));
        THROW_HR_IF(E_FAIL, WaitForSingleObject(fenceEvent.get(), INFINITE) != WAIT_OBJECT_0);
    
        std::vector<const std::byte*> outputData;
        int  i = 0;
        for (int i = 0; i < tensors; i++)
        {
            const std::byte* outputBuffer = reinterpret_cast<const std::byte*>(bindings.GetOutputValues()[i].GetTensorRawData());
            outputData.push_back(outputBuffer);
        }

        auto end = std::chrono::high_resolution_clock::now();
        m_inference_duration = end - start;

        if (outputData.size() > 0)
        {
            // Record start
            auto start = std::chrono::high_resolution_clock::now();
            m_preds.clear();

            if (outputData.size() == 1)
                GetPredictions(outputData[0], output_shapes[0]);
            else if (outputData.size() == 2 && output_names[0] == "box_coords" && output_names[1] == "box_scores")
            {
                // mediapipe onnx model?
                GetFaces(outputData, output_shapes);
            }
            else if (outputData.size() == 3)
                GetPredictions(outputData, output_shapes);

            // Readback the raw data from the model, compute the model's predictions, and render the bounding boxes
            {
                PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render predictions");

                // Print some debug information about the predictions
#if 0
                std::stringstream ss;
                Format(ss, "# of predictions: ", m_preds.size(), "\n");

                for (const auto& pred : m_preds)
                {
                    const char* className = YoloV4Constants::c_classes[pred.predictedClass];
                    int xmin = static_cast<int>(std::round(pred.xmin));
                    int ymin = static_cast<int>(std::round(pred.ymin));
                    int xmax = static_cast<int>(std::round(pred.xmax));
                    int ymax = static_cast<int>(std::round(pred.ymax));

                    Format(ss, "  ", className, ": score ", pred.score, ", box (", xmin, ",", ymin, "),(", xmax, ",", ymax, ")\n");
                }
                OutputDebugStringA(ss.str().c_str());
#endif
                commandList->RSSetViewports(1, &viewport);
                commandList->RSSetScissorRects(1, &scissorRect);

                // Draw bounding box outlines
                m_lineEffect->Apply(commandList);
                m_lineBatch->Begin(commandList);
                float label_height = 5.0f;
                float dx = 5.0f;

                for (auto& pred : m_preds)
                {
                    if (pred.predictedClass < 0)
                    {
                        label_height = 1.0f;
                        dx = 2.0f;
                    }


                    m_lineEffect->SetAlpha(0.75f /*pred.score / 5.0*/);
                   
                    //DirectX::XMVECTORF32 White = { { { 0.980392158f, 0.980392158f, 0.980392158f, 1.0f} } }; // #fafafa
                    //DirectX::XMVECTORF32 White = { { { .0f, 0.980392158f, .0f, 1.0f} } }; // #fafafa
                    int col = colors[((pred.predictedClass < 0) ? 0 : pred.predictedClass) % 20];
                    DirectX::XMVECTORF32 White = { { { (col >> 16) / 255.0f, ((col >> 8) & 0xff) / 255.0f, (col&0xff)/255.0f, 1.0f} } }; // #fafafa
                    for (int i = 0; i < 2; i++)
                    {
                        DirectX::XMVECTORF32 White = { { { (col >> 16) / 255.0f, ((col >> 8) & 0xff) / 255.0f, (col & 0xff) / 255.0f, 1.0f} } }; // #fafafa
                        if (i == 1)
                            White = { { { (col >> 16) / 255.0f, ((col >> 8) & 0xff) / 255.0f, (col & 0xff) / 255.0f, 1.0f} } }; // #fafafa
                        {
                            VertexPositionColor upperLeft(SimpleMath::Vector3(pred.xmin, pred.ymin, 0.f), White);
                            VertexPositionColor upperRight(SimpleMath::Vector3(pred.xmax, pred.ymin, 0.f), White);
                            VertexPositionColor lowerRight(SimpleMath::Vector3(pred.xmax, pred.ymin + label_height * dx, 0.f), White);
                            VertexPositionColor lowerLeft(SimpleMath::Vector3(pred.xmin, pred.ymin + label_height * dx, 0.f), White);
                            m_lineBatch->DrawQuad(upperLeft, upperRight, lowerRight, lowerLeft);
                        }

                        {
                            VertexPositionColor upperLeft(SimpleMath::Vector3(pred.xmin, pred.ymin + dx, 0.f), White);
                            VertexPositionColor upperRight(SimpleMath::Vector3(pred.xmin + dx, pred.ymin + dx, 0.f), White);
                            VertexPositionColor lowerRight(SimpleMath::Vector3(pred.xmin + dx, pred.ymax - dx, 0.f), White);
                            VertexPositionColor lowerLeft(SimpleMath::Vector3(pred.xmin, pred.ymax - dx, 0.f), White);
                            m_lineBatch->DrawQuad(upperLeft, upperRight, lowerRight, lowerLeft);
                        }
                        {

                            VertexPositionColor upperLeft(SimpleMath::Vector3(pred.xmin, pred.ymax - dx, 0.f), White);
                            VertexPositionColor upperRight(SimpleMath::Vector3(pred.xmax - dx, pred.ymax - dx, 0.f), White);
                            VertexPositionColor lowerRight(SimpleMath::Vector3(pred.xmax - dx, pred.ymax, 0.f), White);
                            VertexPositionColor lowerLeft(SimpleMath::Vector3(pred.xmin, pred.ymax, 0.f), White);
                            m_lineBatch->DrawQuad(upperLeft, upperRight, lowerRight, lowerLeft);
                        }
                        {
                            VertexPositionColor upperLeft(SimpleMath::Vector3(pred.xmax - dx, pred.ymin + dx, 0.f), White);
                            VertexPositionColor upperRight(SimpleMath::Vector3(pred.xmax, pred.ymin + dx, 0.f), White);
                            VertexPositionColor lowerRight(SimpleMath::Vector3(pred.xmax, pred.ymax, 0.f), White);
                            VertexPositionColor lowerLeft(SimpleMath::Vector3(pred.xmax - dx, pred.ymax, 0.f), White);
                            m_lineBatch->DrawQuad(upperLeft, upperRight, lowerRight, lowerLeft);
                        }
                    }

                    for (auto p : pred.m_keypoints)
                    {
                        DirectX::XMVECTORF32 KeyColor = { {  {0.0f, 1.0f, .0f, 1.0f} } }; // # green
                        float dx = 3.0f;
                        VertexPositionColor upperLeft(SimpleMath::Vector3(p.first - dx, p.second - dx, 0.f), KeyColor);
                        VertexPositionColor upperRight(SimpleMath::Vector3(p.first + dx, p.second - dx, 0.f), KeyColor);
                        VertexPositionColor lowerRight(SimpleMath::Vector3(p.first + dx, p.second + dx, 0.f), KeyColor);
                        VertexPositionColor lowerLeft(SimpleMath::Vector3(p.first - dx, p.second + dx, 0.f), KeyColor);
                        m_lineBatch->DrawQuad(upperLeft, upperRight, lowerRight, lowerLeft);


                    }
                    /*
                       VertexPositionColor upperLeft(SimpleMath::Vector3(pred.xmin, pred.ymin, 0.f), White);
                        VertexPositionColor upperRight(SimpleMath::Vector3(pred.xmax, pred.ymin, 0.f), White);
                        VertexPositionColor lowerLeft(SimpleMath::Vector3(pred.xmin, pred.ymax, 0.f), White);
                        VertexPositionColor lowerRight(SimpleMath::Vector3(pred.xmax, pred.ymax, 0.f), White);
                        m_lineBatch->DrawQuad(upperLeft, upperRight, lowerRight, lowerLeft);
                     */
                    //m_lineBatch->DrawLine(upperLeft, upperRight);
                    //m_lineBatch->DrawLine(upperRight, lowerRight);
                    //m_lineBatch->DrawLine(lowerRight, lowerLeft);
                    //m_lineBatch->DrawLine(lowerLeft, upperLeft);



                    
                }
                m_lineBatch->End();

                // Draw predicted class labels
                m_spriteBatch->Begin(commandList);
                for (const auto& pred : m_preds)
                {
                    if (pred.predictedClass >= 0)
                    {
                        const char* classText = YoloV4Constants::c_classes[pred.predictedClass];
                        std::wstring classTextW(classText, classText + strlen(classText));

                        // Render a drop shadow by drawing the text twice with a slight offset.
                        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
                            classTextW.c_str(), SimpleMath::Vector2(pred.xmin, pred.ymin - 1.5f * dx) + SimpleMath::Vector2(2.f, 2.f), SimpleMath::Vector4(0.0f, 0.0f, 0.0f, 0.25f));
                        DX::DrawControllerString(m_spriteBatch.get(), m_legendFont.get(), m_ctrlFont.get(),
                            classTextW.c_str(), SimpleMath::Vector2(pred.xmin, pred.ymin - 1.5f * dx), ATG::Colors::DarkGrey);
                    }
                }
                m_spriteBatch->End();

                PIXEndEvent(commandList);
            }
            auto end = std::chrono::high_resolution_clock::now();
            m_output_duration = end - start;
        }
    }

    // Show the new frame.
    PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");

    m_deviceResources->Present();

    PIXEndEvent(m_deviceResources->GetCommandQueue());

    m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());
}


// Helper method to clear the back buffers.
void Sample::Clear()
{
    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Clear");

    // Clear the views.
    auto rtvDescriptor = m_deviceResources->GetRenderTargetView();

    commandList->OMSetRenderTargets(1, &rtvDescriptor, FALSE, nullptr);
    // Use linear clear color for gamma-correct rendering.
    commandList->ClearRenderTargetView(rtvDescriptor, ATG::ColorsLinear::Background, 0, nullptr);

    // Set the viewport and scissor rect.
    auto viewport = m_deviceResources->GetScreenViewport();
    auto scissorRect = m_deviceResources->GetScissorRect();
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissorRect);

    PIXEndEvent(commandList);
}
#pragma endregion

#pragma region Message Handlers
// Message handlers
void Sample::OnActivated()
{
}

void Sample::OnDeactivated()
{
}

void Sample::OnSuspending()
{
}

void Sample::OnResuming()
{
    m_timer.ResetElapsedTime();
    m_gamePadButtons.Reset();
    m_keyboardButtons.Reset();
}

void Sample::OnWindowMoved()
{
    auto r = m_deviceResources->GetOutputSize();
    m_deviceResources->WindowSizeChanged(r.right, r.bottom);
}

void Sample::OnWindowSizeChanged(int width, int height)
{
    if (!m_deviceResources->WindowSizeChanged(width, height))
        return;

    CreateWindowSizeDependentResources();
}

// Properties
void Sample::GetDefaultSize(int& width, int& height) const
{
    width = 1920;
    height = 1080;
}
#pragma endregion

#pragma region Direct3D Resources
// These are the resources that depend on the device.
void Sample::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

    // Create descriptor heaps.
    {
        m_SRVDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            e_srvDescCount);

        m_fontDescriptorHeap = std::make_unique<DescriptorHeap>(
            device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            e_fontDescCount);
    }

   
    CreateTextureResources();
   
    CreateUIResources();
}

void Sample::CreateTextureResources()
{
    auto device = m_deviceResources->GetD3DDevice();
        
    // Create root signatures with one sampler and one texture--one for nearest neighbor sampling,
    // and one for bilinear.
    {
        CD3DX12_DESCRIPTOR_RANGE descRange = {};
        descRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

        CD3DX12_ROOT_PARAMETER rp = {};
        rp.InitAsDescriptorTable(1, &descRange, D3D12_SHADER_VISIBILITY_PIXEL);

        // Nearest neighbor sampling
        D3D12_STATIC_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.MaxAnisotropy = 16;
        samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        samplerDesc.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
        samplerDesc.MinLOD = 0;
        samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
        samplerDesc.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
        rootSignatureDesc.Init(1, &rp, 1, &samplerDesc,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS);

        ComPtr<ID3DBlob> signature;
        ComPtr<ID3DBlob> error;
        HRESULT hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                OutputDebugStringA(reinterpret_cast<const char*>(error->GetBufferPointer()));
            }
            throw DX::com_exception(hr);
        }

        DX::ThrowIfFailed(
            device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                IID_PPV_ARGS(m_texRootSignatureNN.ReleaseAndGetAddressOf())));

        // Bilinear sampling
        samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        rootSignatureDesc.Init(1, &rp, 1, &samplerDesc,
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS);

        hr = D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                OutputDebugStringA(reinterpret_cast<const char*>(error->GetBufferPointer()));
            }
            throw DX::com_exception(hr);
        }

        DX::ThrowIfFailed(
            device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                IID_PPV_ARGS(m_texRootSignatureLinear.ReleaseAndGetAddressOf())));
    }

    // Create the pipeline state for a basic textured quad render, which includes loading shaders.

    {
        auto vertexShaderBlob = DX::ReadData(L"VertexShader.cso");
        auto pixelShaderBlob = DX::ReadData(L"PixelShader.cso");

        static const D3D12_INPUT_ELEMENT_DESC s_inputElementDesc[2] =
        {
            { "SV_Position", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  0 },
            { "TEXCOORD",    0, DXGI_FORMAT_R32G32_FLOAT,       0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,  0 },
        };

        // Describe and create the graphics pipeline state objects (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { s_inputElementDesc, _countof(s_inputElementDesc) };
        psoDesc.pRootSignature = m_texRootSignatureNN.Get();
        psoDesc.VS = { vertexShaderBlob.data(), vertexShaderBlob.size() };
        psoDesc.PS = { pixelShaderBlob.data(), pixelShaderBlob.size() };
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.DSVFormat = m_deviceResources->GetDepthBufferFormat();
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = m_deviceResources->GetBackBufferFormat();
        psoDesc.SampleDesc.Count = 1;
        DX::ThrowIfFailed(
            device->CreateGraphicsPipelineState(&psoDesc,
                IID_PPV_ARGS(m_texPipelineStateNN.ReleaseAndGetAddressOf())));

        psoDesc.pRootSignature = m_texRootSignatureLinear.Get();

        DX::ThrowIfFailed(
            device->CreateGraphicsPipelineState(&psoDesc,
                IID_PPV_ARGS(m_texPipelineStateLinear.ReleaseAndGetAddressOf())));
    }

    // Create vertex buffer for full screen texture render.
    {
        static const Vertex s_vertexData[4] =
        {
            { { -1.f, -1.f, 1.f, 1.0f },{ 0.f, 1.f } },
            { { 1.f, -1.f, 1.f, 1.0f },{ 1.f, 1.f } },
            { { 1.f,  1.f, 1.f, 1.0f },{ 1.f, 0.f } },
            { { -1.f,  1.f, 1.f, 1.0f },{ 0.f, 0.f } },
        };

        // Note: using upload heaps to transfer static data like vert buffers is not 
        // recommended. Every time the GPU needs it, the upload heap will be marshalled 
        // over. Please read up on Default Heap usage. An upload heap is used here for 
        // code simplicity and because there are very few verts to actually transfer.
        DX::ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(sizeof(s_vertexData)),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(m_vertexBuffer.ReleaseAndGetAddressOf())));

        // Copy the quad data to the vertex buffer.
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
        DX::ThrowIfFailed(
            m_vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, s_vertexData, sizeof(s_vertexData));
        m_vertexBuffer->Unmap(0, nullptr);

        // Initialize the vertex buffer view.
        m_vertexBufferView.BufferLocation = m_vertexBuffer->GetGPUVirtualAddress();
        m_vertexBufferView.StrideInBytes = sizeof(Vertex);
        m_vertexBufferView.SizeInBytes = sizeof(s_vertexData);
    }

    // Create index buffer.
    {
        static const uint16_t s_indexData[6] =
        {
            3,1,0,
            2,1,3,
        };

        // See note above
        DX::ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(sizeof(s_indexData)),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(m_indexBuffer.ReleaseAndGetAddressOf())));

        // Copy the data to the index buffer.
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
        DX::ThrowIfFailed(
            m_indexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, s_indexData, sizeof(s_indexData));
        m_indexBuffer->Unmap(0, nullptr);

        // Initialize the index buffer view.
        m_indexBufferView.BufferLocation = m_indexBuffer->GetGPUVirtualAddress();
        m_indexBufferView.Format = DXGI_FORMAT_R16_UINT;
        m_indexBufferView.SizeInBytes = sizeof(s_indexData);
    }

#if USE_VIDEO
    // Create video player.
    {
        wchar_t buff[MAX_PATH]; 
        DX::FindMediaFile(buff, MAX_PATH, c_videoPath);

        m_player = std::make_unique<MediaEnginePlayer>();
        m_player->Initialize(m_deviceResources->GetDXGIFactory(), device, DXGI_FORMAT_B8G8R8A8_UNORM);
        m_player->SetSource(buff);

        while (!m_player->IsInfoReady())
        {
            SwitchToThread();
        }

        m_player->GetNativeVideoSize(m_origTextureWidth, m_origTextureHeight);
        m_player->SetLoop(true);

        // Create texture to receive video frames.
        CD3DX12_RESOURCE_DESC desc(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            0,
            m_origTextureWidth,
            m_origTextureHeight,
            1,
            1,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            1,
            0,
            D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);

        CD3DX12_HEAP_PROPERTIES defaultHeapProperties(D3D12_HEAP_TYPE_DEFAULT);

        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &defaultHeapProperties,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                nullptr,
                IID_PPV_ARGS(m_videoTexture.ReleaseAndGetAddressOf())));

        DX::ThrowIfFailed(
            device->CreateSharedHandle(
                m_videoTexture.Get(),
                nullptr,
                GENERIC_ALL,
                nullptr,
                &m_sharedVideoTexture));

        CreateShaderResourceView(device, m_videoTexture.Get(), m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));
    }
#else
    // Create static texture.
    {
        auto commandList = m_deviceResources->GetCommandList();
        commandList->Reset(m_deviceResources->GetCommandAllocator(), nullptr);

        ComPtr<ID3D12Resource> textureUploadHeap;
    
        D3D12_RESOURCE_DESC txtDesc = {};
        txtDesc.MipLevels = txtDesc.DepthOrArraySize = 1;
        txtDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        txtDesc.SampleDesc.Count = 1;
        txtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;

        wchar_t buff[MAX_PATH];
        DX::FindMediaFile(buff, MAX_PATH, c_imagePath);

        UINT width, height;
        auto image = LoadBGRAImage(buff, width, height);
        txtDesc.Width = m_origTextureWidth = width;
        txtDesc.Height = m_origTextureHeight = height;

        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                &txtDesc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(m_texture.ReleaseAndGetAddressOf())));

        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(m_texture.Get(), 0, 1);

        // Create the GPU upload buffer.
        DX::ThrowIfFailed(
            device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(textureUploadHeap.GetAddressOf())));

        D3D12_SUBRESOURCE_DATA textureData = {};
        textureData.pData = image.data();
        textureData.RowPitch = static_cast<LONG_PTR>(txtDesc.Width * sizeof(uint32_t));
        textureData.SlicePitch = image.size();

        UpdateSubresources(commandList, m_texture.Get(), textureUploadHeap.Get(), 0, 0, 1, &textureData);
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ));

        // Describe and create a SRV for the texture.
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = txtDesc.Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(m_texture.Get(), &srvDesc, m_SRVDescriptorHeap->GetCpuHandle(e_descTexture));
    
        DX::ThrowIfFailed(commandList->Close());
        m_deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(&commandList));

        // Wait until assets have been uploaded to the GPU.
        m_deviceResources->WaitForGpu();
    }
#endif
}

void Sample::CreateUIResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    
    m_lineBatch = std::make_unique<PrimitiveBatch<VertexPositionColor>>(device);

    RenderTargetState rtState(m_deviceResources->GetBackBufferFormat(), m_deviceResources->GetDepthBufferFormat());
    EffectPipelineStateDescription epd(&VertexPositionColor::InputLayout, CommonStates::AlphaBlend,
        CommonStates::DepthDefault, CommonStates::CullNone, rtState, D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
    m_lineEffect = std::make_unique<BasicEffect>(device, EffectFlags::VertexColor, epd);

    SpriteBatchPipelineStateDescription spd(rtState, &CommonStates::AlphaBlend);
    ResourceUploadBatch uploadBatch(device);
    uploadBatch.Begin();

    m_spriteBatch = std::make_unique<SpriteBatch>(device, uploadBatch, spd);

    wchar_t strFilePath[MAX_PATH] = {};
    DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_30.spritefont");
    m_labelFont = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descLabelFont),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descLabelFont));

    DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_30_Bold.spritefont");
    m_labelFontBold = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descLabelFontBold),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descLabelFontBold));

    DX::FindMediaFile(strFilePath, MAX_PATH, L"SegoeUI_18.spritefont");
    m_legendFont = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descLegendFont),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descLegendFont));

    DX::FindMediaFile(strFilePath, MAX_PATH, L"XboxOneControllerLegendSmall.spritefont");
    m_ctrlFont = std::make_unique<SpriteFont>(device, uploadBatch,
        strFilePath,
        m_fontDescriptorHeap->GetCpuHandle(FontDescriptors::e_descCtrlFont),
        m_fontDescriptorHeap->GetGpuHandle(FontDescriptors::e_descCtrlFont));

    auto finish = uploadBatch.End(m_deviceResources->GetCommandQueue());
    finish.wait();
}

// Allocate all memory resources that change on a window SizeChanged event.
void Sample::CreateWindowSizeDependentResources()
{
    auto viewport = m_deviceResources->GetScreenViewport();

    auto proj = DirectX::SimpleMath::Matrix::CreateOrthographicOffCenter(0.f, static_cast<float>(viewport.Width),
        static_cast<float>(viewport.Height), 0.f, 0.f, 1.f);
    m_lineEffect->SetProjection(proj);

    m_spriteBatch->SetViewport(viewport);
}




void Sample::OnDeviceLost()
{
    m_lineEffect.reset();
    m_lineBatch.reset();
    m_spriteBatch.reset();
    m_labelFont.reset();
    m_labelFontBold.reset();
    m_legendFont.reset();
    m_ctrlFont.reset();
    m_fontDescriptorHeap.reset();

    m_player.reset();

    m_texPipelineStateNN.Reset();
    m_texPipelineStateLinear.Reset();
    m_texRootSignatureNN.Reset();
    m_texRootSignatureLinear.Reset();
    m_tensorRenderPipelineState.Reset();
    m_tensorRenderRootSignature.Reset();
    m_texture.Reset();
    m_videoTexture.Reset();
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();

    m_SRVDescriptorHeap.reset();

    m_computePSO.Reset();
    m_computeRootSignature.Reset();

    m_dmlDevice.Reset();
    m_dmlCommandRecorder.Reset();

    m_modelInput.Reset();
    m_modelSOutput = {};
    m_modelMOutput = {};
    m_modelLOutput = {};
    m_dmlOpInitializer.Reset();
    m_dmlGraph.Reset();
    m_modelTemporaryResource.Reset();
    m_modelPersistentResource.Reset();

    m_dmlDescriptorHeap.reset();

    m_graphicsMemory.reset();
}

void Sample::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion
