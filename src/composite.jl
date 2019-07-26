
export CompositeMod

struct CompositeMod{M} <: ArrayModifier
    modifiers   ::  M
end

const EmptyCompositeMod = CompositeMod{Tuple{}}

# Convenience constructor that packs arguments into a tuple
CompositeMod(mods::ArrayModifier...) = CompositeMod(mods)

modifiers(m::CompositeMod) = m.modifiers
modifiers(m::ArrayModifier) = (m,)

first(m::CompositeMod) = first(m.modifiers)
tail(m::CompositeMod) = CompositeMod(tail(m.modifiers))
tail(m::EmptyCompositeMod) = CompositeMod()


"Combine two modifiers into a single modifier, or a composite modifier."
compose(mod1::ArrayModifier, mod2::ArrayModifier) =
    CompositeMod(modifiers(mod2)..., modifiers(mod1)...)


# Provide additional  constructors for a ModifiedArray

modify(A, mods::ArrayModifier...) = modify(A, CompositeMod(mods))
modify(A::ModifiedArray, mod::ArrayModifier) =
    modify(parent(A), compose(modifier(A), mod))
modify(A::ModifiedArray, mods::ArrayModifier...) =
    modify(parent(A), compose(modifier(A), CompositeMod(mods)))


struct CompositeStyle <: ModStyle end

ModStyle(::CompositeMod, ::Interface) = CompositeStyle()
# to avoid an ambiguity
ModStyle(::CompositeMod, ::IF_similar) = CompositeStyle()

mod_eltype(::CompositeStyle, mod::CompositeMod, A) =
    mod_eltype_composite(first(mod), tail(mod), A)
mod_eltype_composite(mod::ArrayModifier, mods::EmptyCompositeMod, A) =
    mod_eltype(mod, A)
mod_eltype_composite(mod::ArrayModifier, mods, A) =
    mod_eltype_composite(ModStyle(mod, IF_eltype()), mod, mods, A)
mod_eltype_composite(::ModNothing, mod, mods, A) =
    mod_eltype_composite(first(mods), tail(mods), A)
mod_eltype_composite(::ModFinal, mod, mods, A) =
    mod_eltype_final(mod)
function mod_eltype_composite(::ModRecursive, mod, mods, A)
    Z = mod_eltype_composite(first(mods), tail(mods), A)
    mod_eltype_post(mod, Z)
end


mod_ndims(::CompositeStyle, mod::CompositeMod, A) =
    mod_ndims_composite(first(mod), tail(mod), A)
mod_ndims_composite(mod::ArrayModifier, mods::EmptyCompositeMod, A) =
    mod_ndims(mod, A)
mod_ndims_composite(mod::ArrayModifier, mods, A) =
    mod_ndims_composite(ModStyle(mod, IF_ndims()), mod, mods, A)
mod_ndims_composite(::ModNothing, mod, mods, A) =
    mod_ndims_composite(first(mods), tail(mods), A)
mod_ndims_composite(::ModFinal, mod, mods, A) =
    mod_ndims_final(mod)
function mod_ndims_composite(::ModRecursive, mod, mods, A)
    Z = mod_ndims_composite(first(mods), tail(mods), A)
    mod_ndims_post(mod, Z)
end

mod_size(::CompositeStyle, mod::CompositeMod, A) =
    mod_size_composite(first(mod), tail(mod), A)
mod_size_composite(mod::ArrayModifier, mods::EmptyCompositeMod, A) =
    mod_size(mod, A)
mod_size_composite(mod::ArrayModifier, mods, A) =
    mod_size_composite(ModStyle(mod, IF_size()), mod, mods, A)
mod_size_composite(::ModNothing, mod, mods, A) =
    mod_size_composite(first(mods), tail(mods), A)
mod_size_composite(::ModFinal, mod, mods, A) =
    mod_size_final(mod)
function mod_size_composite(::ModRecursive, mod, mods, A)
    Z = mod_size_composite(first(mods), tail(mods), A)
    mod_size_post(mod, Z)
end

mod_getindex(::CompositeStyle, mod::CompositeMod, A, I...) =
    mod_getindex_composite(first(mod), tail(mod), A, I...)
mod_getindex_composite(mod::ArrayModifier, mods::EmptyCompositeMod, A, I...) =
    mod_getindex(mod, A, I...)
mod_getindex_composite(mod::ArrayModifier, mods::CompositeMod, A, I...) =
    mod_getindex_composite(ModStyle(mod, IF_getindex()), mod, mods, A, I...)

mod_getindex_composite(::ModNothing, mod, mods, A, I...) =
    mod_getindex_composite(first(mods), tail(mods), A, I...)
mod_getindex_composite(::ModFinal, mod, mods, A, I...) =
    mod_getindex_final(mod, I...)
function mod_getindex_composite(::ModRecursive, mod, mods, A, I...)
    J = mod_getindex_pre(mod, I...)
    Z = mod_getindex_composite(first(mods), tail(mods), A, J...)
    mod_getindex_post(mod, Z, I...)
end

mod_setindex!(::CompositeStyle, mod::CompositeMod, A, val, I...) =
    mod_setindex!_composite(first(mod), tail(mod), A, val, I...)
mod_setindex!_composite(mod::ArrayModifier, mods::EmptyCompositeMod, A, val, I...) =
    mod_setindex!(mod, A, val, I...)
mod_setindex!_composite(mod::ArrayModifier, mods::CompositeMod, A, val, I...) =
    mod_setindex!_composite(ModStyle(mod, IF_setindex!()), mod, mods, A, val, I...)

mod_setindex!_composite(::ModNothing, mod, mods, A, val, I...) =
    mod_setindex!_composite(first(mods), tail(mods), A, val, I...)
mod_setindex!_composite(::ModFinal, mod, mods, A, val, I...) =
    mod_setindex!_final(mod, val, I...)
function mod_setindex!_composite(::ModRecursive, mod, mods, A, val, I...)
    J = mod_setindex!_pre(mod, val, I...)
    mod_setindex!_composite(first(mods), tail(mods), A, val, J...)
end

mod_axes(::CompositeStyle, mod::CompositeMod, A) =
    mod_axes_composite(first(mod), tail(mod), A)
mod_axes_composite(mod::ArrayModifier, mods::EmptyCompositeMod, A) =
    mod_axes(mod, A)
mod_axes_composite(mod::ArrayModifier, mods, A) =
    mod_axes_composite(ModStyle(mod, IF_axes()), mod, mods, A)
mod_axes_composite(::ModNothing, mod, mods, A) =
    mod_axes_composite(first(mods), tail(mods), A)
mod_axes_composite(::ModFinal, mod, mods, A) =
    mod_axes_final(mod)
function mod_axes_composite(::ModRecursive, mod, mods, A)
    Z = mod_axes_composite(first(mods), tail(mods), A)
    mod_axes_post(mod, Z)
end


ModStyle(::Type{<:CompositeMod}, ::IF_IndexStyle) = CompositeStyle()

Base.first(::Type{CompositeMod{Tuple{}}}) = EmptyCompositeMod
Base.first(::Type{CompositeMod{Tuple{A}}}) where {A} = A
Base.first(::Type{CompositeMod{Tuple{A,B}}}) where {A,B} = A
Base.first(::Type{CompositeMod{Tuple{A,B,C}}}) where {A,B,C} = A

Base.tail(::Type{CompositeMod{Tuple{}}}) = EmptyCompositeMod
Base.tail(::Type{CompositeMod{Tuple{A}}}) where {A} = EmptyCompositeMod
Base.tail(::Type{CompositeMod{Tuple{A,B}}}) where {A,B} = CompositeMod{Tuple{B}}
Base.tail(::Type{CompositeMod{Tuple{A,B,C}}}) where {A,B,C} = CompositeMod{Tuple{B,C}}

mod_IndexStyle(::CompositeStyle, Mod, AA) = mod_IndexStyle_composite(first(Mod), tail(Mod), AA)

mod_IndexStyle_composite(Mod::Type{<:ArrayModifier}, Mods::Type{EmptyCompositeMod}, AA) =
    mod_IndexStyle(Mod, AA)
mod_IndexStyle_composite(Mod::Type{<:ArrayModifier}, Mods::Type{<:CompositeMod}, AA) =
    mod_IndexStyle_composite(ModStyle(Mod, IF_IndexStyle()), Mod, Mods, AA)

mod_IndexStyle_composite(::ModNothing, Mod, Mods, AA) =
    mod_IndexStyle_composite(first(Mods), tail(Mods), AA)
mod_IndexStyle_composite(::ModFinal, Mod, Mods, AA) =
    mod_IndexStyle_final(Mod)
function mod_IndexStyle_composite(::ModRecursive, Mod, Mods, AA)
    Z = mod_IndexStyle_composite(first(Mods), tail(Mods), AA)
    mod_IndexStyle_post(Mod, Z)
end
