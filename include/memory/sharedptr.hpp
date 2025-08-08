#ifndef SHAREDPTR
#define SHAREDPTR

struct SharedPtr
{
    void *ptr;
    unsigned int *counter;

    SharedPtr& operator=(const SharedPtr& other){
        this->ptr = other.ptr;
        this->counter = other.counter;
        *counter++;
    }
};

#endif

