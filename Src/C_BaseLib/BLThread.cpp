//
// $Id: BLThread.cpp,v 1.19 2001-11-01 17:08:41 car Exp $
//

#include <winstd.H>
#include <Profiler.H>
#include <BoxLib.H>
#include <Thread.H>

#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>
#endif

#include <iostream>
#include <limits>

#include <cstdio>
#include <ctime>
#include <cerrno>
#include <cstdlib>

#if defined(BL_OSF1)
extern "C" int usleep (useconds_t);
#endif

namespace
{
    const char*
    the_message_string(const char* file, int line, const char* call, int status = 0)
    {
	//
	// Should be large enough.
	//
	const int DIM = 1024;
	static char buf[DIM];
	if ( status )
	{
	    std::sprintf(buf, "BoxLib Thread Error: File %s, line %d, %s: %s",
			 file, line, call, std::strerror(status));
	}
	else
	{
	    std::sprintf(buf, "BoxLib Thread Error: File %s, line %d, %s",
			 file, line, call);
	}
	buf[DIM-1] = '\0';		// Just to be safe.
	return buf;
    }

ThreadSpecificData<int> ts_tid;
Mutex tid_lock;
int thread_counter = 0;
}

namespace BoxLib
{
    void
    Thread_Error(const char* file, int line, const char* call, int status = 0)
    {
	Error(the_message_string(file, line, call,status));
    }

}

#define THREAD_REQUIRE(x)						\
do									\
{									\
  if ( int status = (x) )						\
    {									\
      BoxLib::Thread_Error(__FILE__, __LINE__, #x, status); 		\
    }									\
}									\
while ( false )

#define THREAD_ASSERT(x)						\
do									\
{									\
  if ( !(x) )								\
    {									\
      BoxLib::Thread_Error(__FILE__,__LINE__,#x );			\
    }									\
}									\
while ( false )

// Posix:
#ifdef BL_THREADS
#ifdef WIN32
class Mutex::Implementation
{
public:
    void lock();
    void unlock();
    bool trylock();
private:
    HANDLE m_mutex;
};

class ConditionVariable::Implementation
: public Mutex::Implementation
{
public:
    void signal();
    void broadcast();
    void wait();
private:
    HANDLE m_cv;
};

#else
//
//Mutex
//
class Mutex::Implementation
{
public:
    Implementation ();
    ~Implementation ();
    void lock ();
    bool trylock ();
    void unlock ();
protected:
    friend class ConditionVariable::Implementation;
    pthread_mutex_t m_mutex;
};

class ConditionVariable::Implementation
    : public Mutex::Implementation
{
public:
    Implementation ();
    ~Implementation ();
    void signal ();
    void broadcast ();
    void wait ();
protected:
    pthread_cond_t m_cv;
};
#endif
#endif

void
Thread::sleep(const BoxLib::Time& spec_)
{
#ifdef WIN32
#elif !defined( BL_USE_NANOSLEEP )
    unsigned long tosleep = spec_.as_long();
    if ( tosleep > std::numeric_limits<unsigned long>::max()/1000000 )
    {
	::sleep(tosleep);
    }
    else
    {
	::usleep(tosleep*1000000);
    }
#else
    BoxLib::Time spec = spec_;
    BoxLib::Time rem;
    while ( int status = ::nanosleep(&spec, &rem) )
    {
	if ( errno != EINVAL )
	{
	    BoxLib::Thread_Error(__FILE__, __LINE__, "nanosleep", errno);
	}
	spec = rem;
    }
#endif
}


//
// Barrier
//

Barrier::Barrier(int i)
    : count(0), n_sleepers(0), releasing(false)
{
    init(i);
}

void
Barrier::init(int i)
{
    THREAD_ASSERT( !releasing );
    THREAD_ASSERT( n_sleepers == 0 );
    count = i;
}

void
Barrier::wait()
{
    BL_PROFILE( BL_PROFILE_THIS_NAME() + "::wait()" );
    bool release = false;
    lock();
    // If previous cycle still releasing, wait
    // THREAD_ASSERT ( !releasing );
    while ( releasing )
    {
	ConditionVariable::wait();
    }
    if ( ++n_sleepers == count )
    {
	release = releasing = true;
    }
    else
    {
	// A poor thread cancelation Site
	Thread::CancelState tmp = Thread::setCancelState(Thread::Disable);
	while ( !releasing )
	{
	    ConditionVariable::wait();
	}
	Thread::setCancelState(tmp);
    }
    if ( --n_sleepers == 0 )
    {
	releasing = false;
	release = true;             // Wake up waiters (if any) for next cycle
    }
    unlock();
    if ( release )
    {
	broadcast();
    }
}


//
// Semaphore
//

Semaphore::Semaphore(int val_)
    : value(val_)
{
}

void
Semaphore::wait()
{
    BL_PROFILE( BL_PROFILE_THIS_NAME() + "::wait()" );
    lock();
    while ( value == 0 )
    {
	ConditionVariable::wait();
    }
    value--;
    unlock();
}

bool
Semaphore::trywait()
{
    lock();
    if ( value == 0 )
    {
	unlock();
	return false;
    }
    value--;
    unlock();
    return true;
}

void
Semaphore::post()
{
    lock();
    value++;
    unlock();
    signal();
}


//
//
//

SemaphoreB::SemaphoreB(int val_)
    : val(val_)
{}

int
SemaphoreB::down()
{
    lock();
    while (val <= 0)
    {
	wait();
    }
    int t = --val;
    unlock();

    return t;
}

int
SemaphoreB::up()
{
    lock();
    int t = ++val;
    unlock();
    signal();
    return t;
}

int
SemaphoreB::decrement()
{
    lock();
    int t = --val;
    unlock();
    return t;
}

int
SemaphoreB::value()
{
    lock();
    int t = val;
    unlock();
    return t;
}

//
// SingleBarrier
//

SingleBarrier::SingleBarrier(int i)
    : count(i), n_posters(0), n_waiters(0), releasing(false)
{
}

void
SingleBarrier::wait()
{
    bool release = false;
    lock();
    n_waiters++;
    while ( !releasing )
    {
	ConditionVariable::wait();
    }
    if ( --n_waiters == 0 )
    {
	releasing = false;
	release = true;             // Wake up waiters (if any) for next cycle
	n_posters=0;
    }
    unlock();
    if ( release )
    {
	broadcast();
    }
}

void
SingleBarrier::post()
{
    bool release = false;
    lock();
    // If previous cycle still releasing, wait
    while ( releasing )
    {
	ConditionVariable::wait();
    }
    if ( ++n_posters == count )
    {
	releasing = true;
	release = true;             // Wake up waiters (if any) for next cycle
    }
    unlock();
    if ( release )
    {
	broadcast();
    }
}


//
//Gate
//

Gate::Gate()
    : closed(true)
{
}

void
Gate::open()
{
    lock();
    closed = false;
    broadcast();
    unlock();
}

void
Gate::close()
{
    lock();
    closed = true;
    unlock();
}

void
Gate::release()
{
    broadcast();
}

void
Gate::wait()
{
    BL_PROFILE( BL_PROFILE_THIS_NAME() + "::wait()" );
    lock();
    while ( closed )
    {
	ConditionVariable::wait();
    }
    unlock();
}


//
// Lock<Semaphore> specialization
//

Lock<Semaphore>::Lock(Semaphore& sem_)
    : sem(sem_)
{
    sem.wait();
}

Lock<Semaphore>::~Lock()
{
    sem.post();
}

#ifdef BL_THREADS

namespace
{
extern "C"
{
    typedef void* (*thr_vpvp)(void*);
    typedef void (*thr_vvp)(void*);
}
}



//
// Thread
//
#ifdef WIN32
#else

int
Thread::max_threads()
{
#ifdef PTHREAD_THREADS_MAX
    return PTHREAD_THREADS_MAX;
#else
    return 64;
#endif
}

void
Thread::exit(void* st)
{
    pthread_exit(st);
}

int
Thread::getID()
{
    int* a = ts_tid.get();
    if ( a == 0 )
    {
	ts_tid.set(a = new int(0));
    }
    return *a;
}

void
Thread::yield()
{
#ifdef _POSIX_PRIORITY_SCHEDULING
    sched_yield();
#endif
}

Thread::CancelState
Thread::setCancelState(CancelState state)
{
    CancelState result;
    int newstate;
    switch ( state )
    {
    case Enable:
	newstate = PTHREAD_CANCEL_ENABLE;
	break;
    case Disable:
	newstate = PTHREAD_CANCEL_DISABLE;
	break;
    }
    int oldstate;
    THREAD_REQUIRE( pthread_setcancelstate(newstate, &oldstate) );
    switch ( oldstate )
    {
    case PTHREAD_CANCEL_ENABLE:
	result = Enable;
	break;
    case PTHREAD_CANCEL_DISABLE:
	result = Disable;
	break;
    }
    return result;
}

#endif


#ifdef WIN32

void
Mutex::Implementation::lock ()
{
    WaitForSingleObject(m_mutex, INFINITE);
}

void
Mutex::Implementation::unlock ()
{
    ReleaseMutex(m_mutex);
}

#else
Mutex::Implementation::Implementation()
{
    THREAD_REQUIRE( pthread_mutex_init(&m_mutex, 0) );
}

Mutex::Implementation::~Implementation()
{
    THREAD_REQUIRE( pthread_mutex_destroy(&m_mutex) );
}

void
Mutex::Implementation::lock()
{
    THREAD_REQUIRE( pthread_mutex_lock(&m_mutex) );
}

bool
Mutex::Implementation::trylock()
{
    int status = pthread_mutex_trylock(&m_mutex);
    if ( status == 0 ) return true;
    if ( status == EBUSY ) return false;
    BoxLib::Thread_Error(__FILE__,__LINE__,"pthread_mutex_trylock(&m_mutex)", status);
    return false;
}

void
Mutex::Implementation::unlock()
{
    THREAD_REQUIRE( pthread_mutex_unlock(&m_mutex) );
}
#endif

Mutex::Mutex()
{
    m_impl = new Implementation;
}

Mutex::~Mutex()
{
    delete m_impl;
}

void
Mutex::lock()
{
    m_impl->lock();
}

bool
Mutex::trylock()
{
    return m_impl->trylock();
}

void
Mutex::unlock()
{
    m_impl->unlock();
}


//
// ConditionVariable
//
#ifdef WIN32
#else
ConditionVariable::Implementation::Implementation()
{
    THREAD_REQUIRE( pthread_cond_init(&m_cv, 0) );
}

ConditionVariable::Implementation::~Implementation()
{
    THREAD_REQUIRE( pthread_cond_destroy(&m_cv) );
}

void
ConditionVariable::Implementation::signal()
{
    THREAD_REQUIRE( pthread_cond_signal(&m_cv) );
}

void
ConditionVariable::Implementation::broadcast()
{
    THREAD_REQUIRE( pthread_cond_broadcast(&m_cv) );
}

void
ConditionVariable::Implementation::wait()
{
    THREAD_REQUIRE( pthread_cond_wait(&m_cv, &m_mutex) );
}
#endif

ConditionVariable::ConditionVariable()
{
    m_impl = new Implementation();
}

ConditionVariable::~ConditionVariable()
{
    delete m_impl;
}

void
ConditionVariable::signal()
{
    m_impl->signal();
}

void
ConditionVariable::broadcast()
{
    m_impl->broadcast();
}

void
ConditionVariable::wait()
{
    m_impl->wait();
}

void
ConditionVariable::lock ()
{
    m_impl->lock();
}

void
ConditionVariable::unlock ()
{
    m_impl->unlock();
}

bool
ConditionVariable::trylock ()
{
    return m_impl->trylock();
}


//
//Thread Specific Data
//

#ifdef WIN32

class ThreadSpecificData<void>::Implementation
{
public:
    Implementation(void (*tsd)(void*));
    ~Implementation();
    void* set (const void* v);
    void* get () const;
private:
    HANDLE m_key;
};

ThreadSpecificData<void>::Implementation::Implementation(void (*tsd)(void*))
{
    m_key = TlsAlloc();
}

void*
ThreadSpecificData<void>::Implementation::set(const void* v)
{
    void* ov = TlsGetValue(m_key);
    TlsSetValue(m_key, v);
    return ov;
}

void
ThreadSpecificData<void>::Implementation::get() const
{
    return TlsGetValue(m_key);
}

#else
class ThreadSpecificData<void>::Implementation
{
public:
    Implementation(void (*tsd)(void*));
    ~Implementation();
    void* set (const void* v);
    void* get () const;
private:
    pthread_key_t m_key;
};


ThreadSpecificData<void>::Implementation::Implementation(void (*tsd)(void*))
{
    //printf("%p: ThreadSpecificData<void>::Implementation::Implementation()\n",this);
    THREAD_REQUIRE( pthread_key_create(&m_key, reinterpret_cast<thr_vvp>(tsd)) );
    THREAD_ASSERT(get() == 0);
}

ThreadSpecificData<void>::Implementation::~Implementation()
{
    //printf("%p: ThreadSpecificData<void>::Implementation::~Implementation()\n",this);
    THREAD_REQUIRE( pthread_key_delete(m_key) );
}

void*
ThreadSpecificData<void>::Implementation::set(const void* v)
{
    //printf("%p: ThreadSpecificData<void>::Implementation::set(%p)\n",this,v);
    void* ov = pthread_getspecific(m_key);
    THREAD_REQUIRE( pthread_setspecific(m_key, v) );
    return ov;
}

void*
ThreadSpecificData<void>::Implementation::get() const
{

    void* v = pthread_getspecific(m_key);
    //printf("%p: ThreadSpecificData<void>::Implementation::get(%p)\n",this,v);
    return v;
}
#endif

ThreadSpecificData<void>::ThreadSpecificData(void (*tsd)(void*))
{
    m_impl = new Implementation(tsd);
}

ThreadSpecificData<void>::~ThreadSpecificData()
{
    delete m_impl;
}

void*
ThreadSpecificData<void>::set(const void* v)
{
    return m_impl->set(v);
}

void*
ThreadSpecificData<void>::get() const
{
    return m_impl->get();
}



struct thr_package
{
    Thread_Function m_func;
    void* m_arg;
};

extern "C" void* thr_func(void* arg_)
{
    thr_package* tp = static_cast<thr_package*>(arg_);
    Thread_Function func = tp->m_func;
    void* arg = tp->m_arg;
    delete tp;
    int* a = new int;
    {
	Lock<Mutex> l(tid_lock);
	*a = ++thread_counter;
    }
    // Initially the TS must be NULL
    THREAD_REQUIRE( ts_tid.set(a) == 0 );
    return (*func)(arg);
}

#ifdef WIN32
#include <process.h>

class FunctionThread::Implementation
{
public:
    Implementation (Thread_Function func_, void* arg_, DetachState st_,
                    int stacksize);
    ~Implementation ();
    void* join() const;
    void detach() const;
private:
    HANDLE m_tid;
    mutable bool m_jod;
};

FunctionThread::Implementation::Implementation(Thread_Function func_, void* arg_, DetachState st_)
{
    m_tid = (HANDLE) _beginthreadex(0,0, reinterpret_cast<unsigned int (*)(void*)>(func_),arg_,0,0);
}

void*
FunctionThread::Implementation::join() const
{
    if ( WaitForSingleObject(m_tid, INFINITE) != WAIT_OBJECT_0 )
    {
	CloseHandle(m_tid);
    }
}
void
FunctionThread::Implementation::detach() const
{
    if ( m_tid != 0)
    {
	CloseHandle(m_tid);
    }
}

#else

class FunctionThread::Implementation
{
public:
    Implementation (Thread_Function func_, void* arg_, DetachState st_,
                    int stacksize);
    ~Implementation ();
    void* join() const;
    void detach() const;
private:
    mutable bool m_jod;
    pthread_t m_tid;
};

FunctionThread::Implementation::Implementation(Thread_Function func_,
                                               void* arg_,
                                               DetachState st_,
                                               int stacksize)
    : m_jod(false)
{
    BL_PROFILE( BL_PROFILE_THIS_NAME() + "::FunctionThread()" );
    pthread_attr_t a;
    THREAD_REQUIRE( pthread_attr_init(&a));

    if (stacksize > PTHREAD_STACK_MIN)
    {
        THREAD_REQUIRE(pthread_attr_setstacksize(&a,stacksize));
    }

    int dstate;
    switch ( st_ )
    {
    case Detached:
	m_jod = true;
	dstate = PTHREAD_CREATE_DETACHED;
	break;
    case Joinable:
	m_jod = false;
	dstate = PTHREAD_CREATE_JOINABLE;
	break;
    }
    THREAD_REQUIRE( pthread_attr_setdetachstate(&a, dstate) );
    thr_package* tp = new thr_package;
    tp->m_func = func_;
    tp->m_arg  = arg_;
    THREAD_REQUIRE( pthread_create(&m_tid, &a, func_, arg_) );
}

FunctionThread::Implementation::~Implementation()
{
    detach();
}

void*
FunctionThread::Implementation::join() const
{
    BL_PROFILE( BL_PROFILE_THIS_NAME() + "::join()" );
    void* ret;
    if ( !m_jod )
    {
	THREAD_REQUIRE( pthread_join(m_tid, &ret) );
	m_jod = true;
    }
    return ret;
}

void
FunctionThread::Implementation::detach() const
{
    BL_PROFILE( BL_PROFILE_THIS_NAME() + "::detach()" );
    if ( !m_jod )
    {
	THREAD_REQUIRE( pthread_detach(m_tid) );
    }
    m_jod = true;
}
#endif

FunctionThread::FunctionThread(Thread_Function func_,
                               void* arg_,
                               DetachState st,
                               int stacksize)
{
    m_impl = new Implementation(func_, arg_, st, stacksize);
}

FunctionThread::~FunctionThread()
{
    delete m_impl;
}

void*
FunctionThread::join() const
{
    return m_impl->join();
}

void
FunctionThread::detach() const
{
    m_impl->detach();
}

#else

void
Thread::exit(void*)
{
    std::exit(0);
}

Thread::CancelState
Thread::setCancelState(CancelState)
{
    return Enable;
}


//
//
//

class ThreadSpecificData<void>::Implementation
{
public:
    Implementation(void (*tsd_)(void*));
    ~Implementation();
    void* set(const void* v_);
    void* get() const;
    void* v;
    void (*tsd)(void*);
};

ThreadSpecificData<void>::Implementation::Implementation(void (*tsd_)(void*))
    : v(0), tsd(tsd_)
{
}

ThreadSpecificData<void>::Implementation::~Implementation()
{
}

void*
ThreadSpecificData<void>::Implementation::set(const void* v_)
{
    return v = const_cast<void*>(v_);
}

void*
ThreadSpecificData<void>::Implementation::get() const
{
    return v;
}


ThreadSpecificData<void>::ThreadSpecificData(void (*tsd_)(void*))
{
  m_impl = new Implementation(tsd_);
}

ThreadSpecificData<void>::~ThreadSpecificData()
{
  delete m_impl;
}

void*
ThreadSpecificData<void>::set(const void* v_)
{
    return m_impl->set(v_);
}

void*
ThreadSpecificData<void>::get() const
{
    return m_impl->get();
}

// FuctioinThread
FunctionThread::FunctionThread(Thread_Function func, void* arg_, DetachState st, int stacksize)
{
    func(arg_);
}

FunctionThread::~FunctionThread()
{
    detach();
}

void*
FunctionThread::join() const
{
    return 0;
}

void
FunctionThread::detach() const
{
}

#endif

    // #include <process.h>
    // #define pthread_t HANDLE
    // #define pthread_attr_t DWORD
    // #define pthread_create(thhandle,attr,thfunc,tharg)
    // ((thhandle=(HANDLE)_beginthreadex(NULL,0,thfunc,tharg,0, NULL))==NULL)
    // #define pthread_self() GetCurrentThreadId()
    // #define THREAD_FUNCTION void *
    // #define THREAD_FUNCTION_RETURN void *
    // #define THREAD_SPECIFIC_INDEX pthread_key_t
    // #define ts_key_create(ts_key, destructor) pthread_key_create (&(ts_key), destructor);
/* Syncrhronization macros - Assume NT 4.0 or 2000 */
    // #define pthread_mutex_t HANDLE
    // #define pthread_condvar_t HANDLE
    // #define CV_TIMEOUT 100 /* Arbitrary value */
/* USE THE FOLLOWING FOR WINDOWS 9X */
//#define cond_var_wait(cv,mutex) {
//ReleaseMutex(mutex);
//WaitForSingleObject(cv,CV_TIMEOUT);
//WaitForSingleObject(mutex,INFINITE);
//};*/
// #define pthread_cond_wait(cv,mutex) {
//SignalObjectAndWait(mutex,cv,INFINITE,FALSE);
//WaitForSingleObject(mutex,INFINITE);
//};
// #define pthread_cond_broadcast(cv) PulseEvent(cv)
