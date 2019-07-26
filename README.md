# ModifiedArrays.jl
A generic package to modify the behaviour of Julia arrays using composition with modifier objects.

There are various existing packages that modify an `AbstractArray` to provide new functionality. There are several examples of this in the [JuliaArrays](https://github.com/JuliaArrays) collection. Julia Base itself implements the `Adjoint` and `Transpose` types, to lazily represent the adjoint or transpose of a matrix.

Such modifications of arrays are usually implemented via inheritance. The new type inherits from `AbstractArray` and holds a pointer to the `parent` array. The functionality is implemented by overriding the interface of `AbstractArray` for the new type, and invoking the methods defined for the parent array with some modifications.

The goal of this package is to explore an alternative design pattern and to replace inheritance with composition. The modified array holds two pointers, one to the `parent` array and one to a `Modifier`. This approach has advantages and disadvantages.

The key differences include:

- The modifier can be developed independently of the array. Hence, the modifiers are reusable: they can be combined with any object that implements the `AbstractArray` interface, even if that object itself doesn't inherit from `AbstractArray`.
- Modifiers can easily be combined. An array that is modified twice contains a single parent array and a tuple of two modifiers, as opposed to an inheritance tree with two layers and a parent pointer each.
- Modifiers can be structured into groups that enable more code reuse. For example, a matrix can be declared to be diagonal by adding a property modifier that only changes the outcome of `isdiag`. This takes about four to five lines of extra code in total.
- Arrays with a tuple of modifiers are implemented using recursion. The recursion tree is functionally equivalent to the inheritance tree of inheritance-based modification.

Modifiers that are included in the package reproduce the functionality of the array types `Adjoint`, `Transpose` and `OffsetArray`. The implementation passes the complete test suite of these types (with minor modifications related mostly to syntax).

This package is not optimized for performance. It is unlikely for the `OffsetArray` in this package to be faster than the original [`OffsetArrays`](https://github.com/JuliaArrays/OffsetArrays.jl). On the other hand, it seems likely for this package to eventually result in faster code for the transpose of a view of an `OffsetArray`.
