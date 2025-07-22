#include "datatypes.hpp"
#include "logger.cuh"
#include <cublas_v2.h>
#include <vector>

namespace IdioticML {

/**
 * ADDITION
 */
inline void CPUtensorAdd_helper(void *dest_tensor, 
                                const void *src_tensor1, 
                                const void *src_tensor2,
                                unsigned int numberOfEntries,
                                DataType::DataType datatype)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((float *)dest_tensor)[i] = ((float *)src_tensor1)[i] + ((float *)src_tensor2)[i];
        }
        break;
    }
    case DataType::DataType::DOUBLE: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((double *)dest_tensor)[i] = ((double *)src_tensor1)[i] + ((double *)src_tensor2)[i];
        }
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}

inline void GPUtensorAdd_helper(void *dest_tensor, 
                                const void *other_tensor, 
                                unsigned int numberOfEntries,
                                DataType::DataType datatype,
                                cublasHandle_t handler)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        float alpha = 1;
        CUBLAS_CHECK(cublasSaxpy(handler, 
                                 numberOfEntries, 
                                 &alpha, 
                                 (float *)other_tensor, 1, 
                                 (float *)dest_tensor, 1));
        break;
    }
    case DataType::DataType::DOUBLE: {
        double alpha = 1;
        CUBLAS_CHECK(cublasDaxpy(handler, 
                                 numberOfEntries, 
                                 &alpha, 
                                 (double *)other_tensor, 1, 
                                 (double *)dest_tensor, 1));
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}

/**
 * SUBTRACTION
 */
inline void CPUtensorSub_helper(void *dest_tensor, 
                                const void *src_tensor1, 
                                const void *src_tensor2,
                                unsigned int numberOfEntries,
                                DataType::DataType datatype)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((float *)dest_tensor)[i] = ((float *)src_tensor1)[i] - ((float *)src_tensor2)[i];
        }
        break;
    }
    case DataType::DataType::DOUBLE: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((double *)dest_tensor)[i] = ((double *)src_tensor1)[i] - ((double *)src_tensor2)[i];
        }
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}
inline void GPUtensorSub_helper(void *dest_tensor, 
                                const void *other_tensor, 
                                unsigned int numberOfEntries,
                                DataType::DataType datatype,
                                cublasHandle_t handler)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        float alpha = -1;
        CUBLAS_CHECK(cublasSaxpy(handler, 
                                 numberOfEntries, 
                                 &alpha, 
                                 (float *)other_tensor, 1, 
                                 (float *)dest_tensor, 1));
        break;
    }
    case DataType::DataType::DOUBLE: {
        double alpha = -1;
        CUBLAS_CHECK(cublasDaxpy(handler, 
                                 numberOfEntries, 
                                 &alpha, 
                                 (double *)other_tensor, 1, 
                                 (double *)dest_tensor, 1));
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}


/**
 * MATRIX MULTIPLICATION
 */
inline void GPUmatrixMult_helper(const std::vector<unsigned int>& thisDimensions,
                                 const std::vector<unsigned int>& otherDimensions,
                                 const void* thisTensorPtr,
                                 const void* otherTensorPtr,
                                 void *newTensorPtr,
                                 DataType::DataType type,
                                 cublasHandle_t handler)
{
    switch (type)
    {
    case DataType::DataType::FLOAT:{
        float a = 1;
        float b = 0;
        CUBLAS_CHECK(cublasSgemm(handler,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 otherDimensions[0], thisDimensions[1], thisDimensions[0],
                                 &a,
                                 (float *)otherTensorPtr, otherDimensions[0],
                                 (float *)thisTensorPtr,  thisDimensions[0],
                                 &b,
                                 (float *)newTensorPtr, otherDimensions[0]));   
        break;
    }
    case DataType::DataType::DOUBLE: {
        double a = 1;
        double b = 0;
        CUBLAS_CHECK(cublasDgemm(handler,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 otherDimensions[0], thisDimensions[1], thisDimensions[0],
                                 &a,
                                 (double *)otherTensorPtr, otherDimensions[0],
                                 (double *)thisTensorPtr,  thisDimensions[0],
                                 &b,
                                 (double *)newTensorPtr, otherDimensions[0]));  
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}





}