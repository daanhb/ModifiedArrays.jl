
abstract type ViewModifier{I} <: IndexingModifier end

struct NonmutatingViewModifier{I} <: ViewModifier{I}
    indices ::  I
end

struct MutatingViewModifier{I} <: ViewModifier{I}
    indices ::  I
end

modified_size(mod::ViewModifier, A) = size(mod.indices)
modified_size(mod::ViewModifier, A, d) = size(mod.indices, d)

transformindex(mod::ViewModifier, A, I) = mod.indices[I]
