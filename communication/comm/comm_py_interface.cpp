#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "common.h"
#include "comm.h"

struct PyComm{
    PyObject_HEAD
    /* Type-specific fields go here. */
    Comm* internal;
};

static int PyCommInit(PyComm *self, PyObject *args, PyObject *kwds)
{
    char *kwlist[] = {
        (char *)"nrank",
        (char*)"rank",
        (char*)"device",
        nullptr
    };
    int nrank, rank, device;
    PYARGCHECK(PyArg_ParseTupleAndKeywords(
        args, kwds, "|iii", kwlist,
        &nrank, &rank, &device
    ));
    if (self->internal != nullptr) delete self->internal;
    self->internal = new Comm(nrank, rank, device);
    return 0;
}

static void PyCommDealloc(PyComm *self)
{
    if (self->internal != nullptr) delete self->internal;
}

static PyObject* getUniqueId(PyComm *self, PyObject *args)
{
    const char* res = self->internal->getUniqueId();
    return PyByteArray_FromStringAndSize(res, 128);
}

static PyObject* setUniqueId(PyComm *self, PyObject *args)
{
    Py_buffer buf;
    PYARGCHECK(PyArg_ParseTuple(args, "y*" , &buf));
    self->internal->setUniqueId((const char*)buf.buf);
    Py_RETURN_NONE;
}

static PyObject* commInitRank(PyComm *self, PyObject *args) {
    int nrank, rank;
    Py_BEGIN_ALLOW_THREADS;
    self->internal->commInitRank();
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject* commDestroy(PyComm *self, PyObject *args) {
    self->internal->commDestroy();
    Py_RETURN_NONE;
}

static PyObject* syncStream(PyComm *self, PyObject *args) {
    Py_BEGIN_ALLOW_THREADS;
    self->internal->syncStream();
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject* send(PyComm *self, PyObject *args) {
    void* data;
    ssize_t N;
    ncclDataType_t dtype;
    int peer;
    PYARGCHECK(PyArg_ParseTuple(args, "llii" , &data, &N, &dtype, &peer));
    Py_BEGIN_ALLOW_THREADS;
    self->internal->send(data, N, dtype, peer);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyObject* recv(PyComm *self, PyObject *args) {
    void* data;
    ssize_t N;
    ncclDataType_t dtype;
    int peer;
    PYARGCHECK(PyArg_ParseTuple(args, "llii" , &data, &N, &dtype, &peer));
    Py_BEGIN_ALLOW_THREADS;
    self->internal->recv(data, N, dtype, peer);
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

static PyMethodDef PyCommMethods[] = {
    {"getUniqueId", (PyCFunction)getUniqueId, METH_VARARGS, "get nccl unique id"},
    {"setUniqueId", (PyCFunction)setUniqueId, METH_VARARGS, "set nccl unique id"},
    {"send", (PyCFunction)send, METH_VARARGS, "sending buffer to peer"},
    {"recv", (PyCFunction)recv, METH_VARARGS, "receiving buffer from peer"},
    {"syncStream", (PyCFunction)syncStream, METH_VARARGS, "Sync stream"},
    {"commInitRank", (PyCFunction)commInitRank, METH_VARARGS, "Initialize communication"},
    {"commDestroy", (PyCFunction)commDestroy, METH_VARARGS, "End communication"},
    {NULL, NULL, 0, NULL}
};

PyTypeObject PyCommType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "PyComm",                                    /* tp_name */
  sizeof(PyComm),                              /* tp_basicsize */
  0,                                           /* tp_itemsize */
  (destructor)PyCommDealloc,                   /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                          /* tp_flags */
  nullptr,                                     /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  PyCommMethods,                               /* tp_methods */
  nullptr,                                     /* tp_members */
  nullptr,                                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  (initproc)PyCommInit,                        /* tp_init */
  nullptr,                                     /* tp_alloc */
  PyType_GenericNew,                           /* tp_new */
};

static struct PyModuleDef ncclmodule = {
    PyModuleDef_HEAD_INIT,
    "nccl",
    NULL,
    -1,
};

PyMODINIT_FUNC
PyInit_nccl(void)
{
    PyObject *m;
    if (PyType_Ready(&PyCommType) < 0)
        return NULL;

    m = PyModule_Create(&ncclmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PyCommType);
    if (PyModule_AddObject(m, "PyComm", (PyObject *) &PyCommType) < 0) {
        Py_DECREF(&PyCommType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}