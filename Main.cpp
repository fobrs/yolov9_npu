//--------------------------------------------------------------------------------------
// Main.cpp
//
// Entry point for Windows desktop application.
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//--------------------------------------------------------------------------------------

#include "pch.h"
#include "resource.h"

#include "yolov9npu.h"

#include <wingdi.h>
#include <commdlg.h >
#include "FileDialog.h"

using namespace DirectX;

namespace
{
    std::unique_ptr<Sample> g_sample;
};

LPCWSTR g_szAppName = L"yolov9npu";

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Indicates to hybrid graphics systems to prefer the discrete part by default
extern "C"
{
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

// Entry point
int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    //UNREFERENCED_PARAMETER(lpCmdLine);

    if (!XMVerifyCPUSupport())
        return 1;

    Microsoft::WRL::Wrappers::RoInitializeWrapper initialize(RO_INIT_MULTITHREADED);
    if (FAILED(initialize))
        return 1;

    g_sample = std::make_unique<Sample>();

    // Register class and create window
    {
        // Register class
        WNDCLASSEX wcex;
        wcex.cbSize = sizeof(WNDCLASSEX);
        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = WndProc;
        wcex.cbClsExtra = 0;
        wcex.cbWndExtra = 0;
        wcex.hInstance = hInstance;
        wcex.hIcon = LoadIcon(hInstance, L"IDI_ICON");
        wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wcex.hbrBackground = (HBRUSH) (COLOR_WINDOW + 1);
        wcex.lpszMenuName = nullptr;
        wcex.lpszClassName = L"yoloWindowClass";
        wcex.hIconSm = LoadIcon(wcex.hInstance, L"IDI_ICON");
        if (!RegisterClassEx(&wcex))
            return 1;

        // Create window
        int w, h;
        g_sample->GetDefaultSize(w, h);

        RECT rc;
        rc.top = 0;
        rc.left = 0;
        rc.right = static_cast<LONG>(w); 
        rc.bottom = static_cast<LONG>(h);

        AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

        HWND hwnd = CreateWindowEx(0, L"yoloWindowClass", g_szAppName, WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, hInstance,
            nullptr);
        // Change to CreateWindowEx(WS_EX_TOPMOST, L"yolov4WindowClass", g_szAppName, WS_POPUP,
        // to default to fullscreen.

        if (!hwnd)
            return 1;

        ShowWindow(hwnd, nCmdShow);
        // Change nCmdShow to SW_SHOWMAXIMIZED to default to fullscreen.

        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(g_sample.get()) );

        GetClientRect(hwnd, &rc);
        auto run_on_gpu = false;
        if (wcsstr(lpCmdLine, L"gpu") > 0)
            run_on_gpu = true;

        if (!g_sample->Initialize(hwnd, rc.right - rc.left, rc.bottom - rc.top, run_on_gpu))
            return 1;
    }

    // Main message loop
    MSG msg = {};
    while (WM_QUIT != msg.message)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            g_sample->Tick();
        }
    }

    g_sample.reset();

    ExitProcess(1);
    // CoUninitialize() hangs!!
    CoUninitialize();

    return (int) msg.wParam;
}

// Windows procedure
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hdc;

    static bool s_in_sizemove = false;
    static bool s_in_suspend = false;
    static bool s_minimized = false;
    static bool s_fullscreen = false;
    // Set s_fullscreen to true if defaulting to fullscreen.

    auto sample = reinterpret_cast<Sample*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));

    switch (message)
    {
    case WM_PAINT:
        if (s_in_sizemove && sample)
        {
            sample->Tick();
        }
        else
        {
            hdc = BeginPaint(hWnd, &ps);
            EndPaint(hWnd, &ps);
        }
        break;

    case WM_MOVE:
        if (sample)
        {
            sample->OnWindowMoved();
        }
        break;

    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
        {
            if (!s_minimized)
            {
                s_minimized = true;
                if (!s_in_suspend && sample)
                    sample->OnSuspending();
                s_in_suspend = true;
            }
        }
        else if (s_minimized)
        {
            s_minimized = false;
            if (s_in_suspend && sample)
                sample->OnResuming();
            s_in_suspend = false;
        }
        else if (!s_in_sizemove && sample)
        {
            sample->OnWindowSizeChanged(LOWORD(lParam), HIWORD(lParam));
        }
        break;

    case WM_ENTERSIZEMOVE:
        s_in_sizemove = true;
        break;

    case WM_EXITSIZEMOVE:
        s_in_sizemove = false;
        if (sample)
        {
            RECT rc;
            GetClientRect(hWnd, &rc);

            sample->OnWindowSizeChanged(rc.right - rc.left, rc.bottom - rc.top);
        }
        break;

    case WM_GETMINMAXINFO:
    {
        auto info = reinterpret_cast<MINMAXINFO*>(lParam);
        info->ptMinTrackSize.x = 320;
        info->ptMinTrackSize.y = 200;
    }
    break;

    case WM_ACTIVATEAPP:
        if (sample)
        {
            Keyboard::ProcessMessage(message, wParam, lParam);
            Mouse::ProcessMessage(message, wParam, lParam);

            if (wParam)
            {
                sample->OnActivated();
            }
            else
            {
                sample->OnDeactivated();
            }
        }
        break;

    case WM_POWERBROADCAST:
        switch (wParam)
        {
        case PBT_APMQUERYSUSPEND:
            if (!s_in_suspend && sample)
                sample->OnSuspending();
            s_in_suspend = true;
            return TRUE;

        case PBT_APMRESUMESUSPEND:
            if (!s_minimized)
            {
                if (s_in_suspend && sample)
                    sample->OnResuming();
                s_in_suspend = false;
            }
            return TRUE;
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    case WM_LBUTTONDBLCLK:
    {
        volatile int a = 0;
        break;
    }
    case WM_INPUT:
    case WM_MOUSEMOVE:
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_RBUTTONDOWN:
        Mouse::ProcessMessage(message, wParam, lParam);
        break;
    case WM_RBUTTONUP:
    case WM_CONTEXTMENU:
    {
        RECT rect;
        POINT p = { GET_X_LPARAM(lParam) , GET_Y_LPARAM(lParam) };
        //ScreenToClient(hWnd, &p);
        GetClientRect(hWnd, &rect);
       
        std::wstring title = L"Open Video file";
        std::wstring filename;
        bool bVideo = true;

        // where did I click?
        bool bAddModel = false;
        if (p.y > rect.top + ((rect.bottom - rect.top) * 9 / 10))
        {
            bVideo = false;
            if ((GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0)
            {
                title = L"Add ONNX model";
                bAddModel = true;
            }
            else
                 title = L"Open ONNX model";           
        }

        auto hr = FileOpen(title, filename, bVideo, bAddModel);
        if (hr == S_OK)
        
        //if (GetOpenFileName(&ofn) == TRUE)
        {
            //
            // use ofn.lpstrFile
            
            if (!bVideo)
            {
                sample->OnNewMopdel(filename.c_str(), bAddModel);
            }
            else
            {
                sample->OnNewFile(filename.c_str());
            }          
        }
        else
        {
            sample->m_player->Skip(-5);

        }
        return 1;
    break;
    }
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
    case WM_MOUSEWHEEL:
    case WM_XBUTTONDOWN:
    case WM_XBUTTONUP:
    case WM_MOUSEHOVER:
        Mouse::ProcessMessage(message, wParam, lParam);
        break;

    case WM_KEYDOWN:
    case WM_KEYUP:
    case WM_SYSKEYUP:
        Keyboard::ProcessMessage(message, wParam, lParam);
        break;

    case WM_SYSKEYDOWN:
        if (wParam == VK_RETURN && (lParam & 0x60000000) == 0x20000000)
        {
            // Implements the classic ALT+ENTER fullscreen toggle
            if (s_fullscreen)
            {
                SetWindowLongPtr(hWnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);
                SetWindowLongPtr(hWnd, GWL_EXSTYLE, 0);

                int width = 800;
                int height = 600;
                if (sample)
                    sample->GetDefaultSize(width, height);

                ShowWindow(hWnd, SW_SHOWNORMAL);

                SetWindowPos(hWnd, HWND_TOP, 0, 0, width, height, SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED);
            }
            else
            {
                SetWindowLongPtr(hWnd, GWL_STYLE, 0);
                SetWindowLongPtr(hWnd, GWL_EXSTYLE, WS_EX_TOPMOST);

                SetWindowPos(hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);

                ShowWindow(hWnd, SW_SHOWMAXIMIZED);
            }

            s_fullscreen = !s_fullscreen;
        }
        Keyboard::ProcessMessage(message, wParam, lParam);
        break;

    case WM_MENUCHAR:
        // A menu is active and the user presses a key that does not correspond
        // to any mnemonic or accelerator key. Ignore so we don't produce an error beep.
        return MAKELRESULT(0, MNC_CLOSE);
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

// Exit helper
void ExitSample()
{
    PostQuitMessage(0);
}
