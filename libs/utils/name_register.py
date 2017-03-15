# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

__author__ = 'fyabc'


class NameRegister(object):
    """A helper class to select different class by string names or aliases in options.

    Usage:
        # Library code
        class SomeBase(NameRegister):
            NameTable = {}
            ...

        class Some1(SomeBase):
            ...

        Some1.register_class(['name1', 'name2', ...])

        # User code
        SomeBase.get_by_name('name1')   # => class Some1
    """

    # [NOTE] Subclasses MUST override this variable.
    NameTable = {}

    @classmethod
    def register_class(cls, aliases, clazz=None):
        clazz = cls if clazz is None else clazz

        # Default add class name (lower)
        cls.NameTable[clazz.__name__.lower()] = clazz
        for name in aliases:
            cls.NameTable[name] = clazz

    @classmethod
    def get_by_name(cls, name):
        return cls.NameTable[name.lower()]
