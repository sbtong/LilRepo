import new
import inspect

def _getattr(obj, name):
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return None

def _setattr(obj, name, val):
    object.__setattr__(obj, name, val)

def _proto_getattr(obj, name):
    val = _getattr(obj, name)
    if val is None:
        parent = _getattr(obj, '__proto__')
        val = _getattr(parent, name)
    return val

class ObjectMetaClass(type):
    def __repr__(self):
        return "<constructor '%s'>" % self.__name__

class Object(object):
    __metaclass__ = ObjectMetaClass
    prototype = None
    
    def __init__(self):
        self.__proto__ = self.prototype
        self.constructor = self.__class__
    
    def __getattribute__(self, name):
        val = _proto_getattr(self, name)
        if isinstance(val, property) and val.fget:
            get = new.instancemethod(val.fget, self)
            return get()
        elif inspect.isfunction(val):
            func = new.instancemethod(val, self)
            return func
        else:
            return val
            
    def __setattr__(self, name, val):
        if not isinstance(val, property):
            _val = _proto_getattr(self, name)
            if isinstance(_val, property) and _val.fset:
                _val.fset(self, val)
                return
        _setattr(self, name, val)

    def __delattr__(self, name):
        val = _proto_getattr(self, name)
        if isinstance(val, property) and val.fdel:
            val.fdel(self)
        else:
            object.__delattr__(self, name)

Object.prototype = Object()

def constructor(func):
    ret = type(func.__name__, (Object,), dict())
    ret.prototype = ret()
    def init(self, *vargs, **kwargs):
        Object.__init__(self)
        func(self, *vargs, **kwargs)
    ret.__init__ = init
    return ret