#include <CL/cl.h>
#include <stdio.h>
#include <string.h>

int main() {
    cl_uint n;
    cl_platform_id platforms[10];

    cl_int err = clGetPlatformIDs(10, platforms, &n);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs failed: %d\n", err);
        return 1;
    }

    printf("Found %u platform(s)\n", n);

    for (cl_uint i = 0; i < n; i++) {
        char name[128], vendor[128], version[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 128, vendor, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 128, version, NULL);

        printf("Platform %u: %s\n", i, name);
        printf("  Vendor: %s\n", vendor);
        printf("  Version: %s\n", version);

        // Optional: patch the platform name to Mali to test MNN detection
        if (strcmp(name, "ARM Platform") == 0) {
            strcpy(name, "Mali");
            printf("  (Temporarily renamed to: %s)\n", name);
        }

        // List devices
        cl_uint num_devices = 0;
        cl_device_id devices[10];
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);
        if (err != CL_SUCCESS) {
            printf("  No devices found or error: %d\n", err);
            continue;
        }

        for (cl_uint j = 0; j < num_devices; j++) {
            char dev_name[128], dev_vendor[128], dev_ext[1024];
            cl_device_type type;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, dev_name, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 128, dev_vendor, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 1024, dev_ext, NULL);

            printf("  Device %u: %s\n", j, dev_name);
            printf("    Vendor: %s\n", dev_vendor);
            printf("    Type: %s\n", type == CL_DEVICE_TYPE_GPU ? "GPU" : "Other");
            printf("    Extensions: %s\n", dev_ext);
        }
    }

    return 0;
}
