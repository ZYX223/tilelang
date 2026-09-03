#include <Python.h>

#include <stdio.h>

int main(int argc, char **argv) {
    wchar_t **wide_argv = PyMem_RawMalloc((size_t)argc * sizeof(wchar_t *));
    if (wide_argv == NULL) {
        fprintf(stderr, "tilelang_swpython: failed to allocate Python argv\n");
        return 1;
    }

    int converted = 0;
    for (; converted < argc; ++converted) {
        wide_argv[converted] = Py_DecodeLocale(argv[converted], NULL);
        if (wide_argv[converted] == NULL) {
            fprintf(stderr, "tilelang_swpython: failed to decode argument %d\n", converted);
            break;
        }
    }

    if (converted != argc) {
        for (int i = 0; i < converted; ++i) {
            PyMem_RawFree(wide_argv[i]);
        }
        PyMem_RawFree(wide_argv);
        return 2;
    }

    // The -mdynamic executable initializes CRTS before Py_Main imports
    // SWPyTorch or a registration extension from the AOT operator bundle.
    // Keep argv alive until process teardown; Python 3.6 owns interpreter
    // shutdown and the process exits immediately after this call.
    return Py_Main(argc, wide_argv);
}
