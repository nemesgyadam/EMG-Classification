??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
?	?*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_23/gamma
?
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_23/beta
?
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_23/moving_mean
?
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_23/moving_variance
?
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
_output_shapes	
:?*
dtype0
|
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_32/kernel
u
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel* 
_output_shapes
:
??*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_24/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_24/gamma
?
0batch_normalization_24/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_24/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_24/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_24/beta
?
/batch_normalization_24/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_24/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_24/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_24/moving_mean
?
6batch_normalization_24/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_24/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_24/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_24/moving_variance
?
:batch_normalization_24/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_24/moving_variance*
_output_shapes	
:?*
dtype0
{
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d* 
shared_namedense_33/kernel
t
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes
:	?d*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:d*
dtype0
?
batch_normalization_25/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_namebatch_normalization_25/gamma
?
0batch_normalization_25/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_25/gamma*
_output_shapes
:d*
dtype0
?
batch_normalization_25/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_namebatch_normalization_25/beta
?
/batch_normalization_25/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_25/beta*
_output_shapes
:d*
dtype0
?
"batch_normalization_25/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"batch_normalization_25/moving_mean
?
6batch_normalization_25/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_25/moving_mean*
_output_shapes
:d*
dtype0
?
&batch_normalization_25/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*7
shared_name(&batch_normalization_25/moving_variance
?
:batch_normalization_25/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_25/moving_variance*
_output_shapes
:d*
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dF* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:dF*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:F*
dtype0
?
batch_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*-
shared_namebatch_normalization_26/gamma
?
0batch_normalization_26/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_26/gamma*
_output_shapes
:F*
dtype0
?
batch_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*,
shared_namebatch_normalization_26/beta
?
/batch_normalization_26/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_26/beta*
_output_shapes
:F*
dtype0
?
"batch_normalization_26/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*3
shared_name$"batch_normalization_26/moving_mean
?
6batch_normalization_26/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_26/moving_mean*
_output_shapes
:F*
dtype0
?
&batch_normalization_26/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:F*7
shared_name(&batch_normalization_26/moving_variance
?
:batch_normalization_26/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_26/moving_variance*
_output_shapes
:F*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F2* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:F2*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:2*
dtype0
?
batch_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*-
shared_namebatch_normalization_27/gamma
?
0batch_normalization_27/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_27/gamma*
_output_shapes
:2*
dtype0
?
batch_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*,
shared_namebatch_normalization_27/beta
?
/batch_normalization_27/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_27/beta*
_output_shapes
:2*
dtype0
?
"batch_normalization_27/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*3
shared_name$"batch_normalization_27/moving_mean
?
6batch_normalization_27/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_27/moving_mean*
_output_shapes
:2*
dtype0
?
&batch_normalization_27/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*7
shared_name(&batch_normalization_27/moving_variance
?
:batch_normalization_27/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_27/moving_variance*
_output_shapes
:2*
dtype0
z
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
* 
shared_namedense_36/kernel
s
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes

:2
*
dtype0
r
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_36/bias
k
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes
:
*
dtype0
?
batch_normalization_28/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namebatch_normalization_28/gamma
?
0batch_normalization_28/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_28/gamma*
_output_shapes
:
*
dtype0
?
batch_normalization_28/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namebatch_normalization_28/beta
?
/batch_normalization_28/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_28/beta*
_output_shapes
:
*
dtype0
?
"batch_normalization_28/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"batch_normalization_28/moving_mean
?
6batch_normalization_28/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_28/moving_mean*
_output_shapes
:
*
dtype0
?
&batch_normalization_28/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&batch_normalization_28/moving_variance
?
:batch_normalization_28/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_28/moving_variance*
_output_shapes
:
*
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

:
*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?^
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?^
value?]B?] B?]
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer-15
layer_with_weights-12
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
?
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'trainable_variables
(	variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
?
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
?
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
R
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
h

\kernel
]bias
^trainable_variables
_	variables
`regularization_losses
a	keras_api
?
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
h

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
l

~kernel
bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
&
	?iter

?decay
?momentum
?
0
1
#2
$3
+4
,5
26
37
:8
;9
A10
B11
I12
J13
P14
Q15
\16
]17
c18
d19
k20
l21
r22
s23
~24
25
?
0
1
#2
$3
%4
&5
+6
,7
28
39
410
511
:12
;13
A14
B15
C16
D17
I18
J19
P20
Q21
R22
S23
\24
]25
c26
d27
e28
f29
k30
l31
r32
s33
t34
u35
~36
37
 
?
trainable_variables
?metrics
?non_trainable_variables
	variables
?layer_metrics
?layers
regularization_losses
 ?layer_regularization_losses
 
 
 
 
?
trainable_variables
?metrics
?non_trainable_variables
	variables
?layer_metrics
?layers
regularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
?metrics
?non_trainable_variables
	variables
?layer_metrics
?layers
 regularization_losses
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
%2
&3
 
?
'trainable_variables
?metrics
?non_trainable_variables
(	variables
?layer_metrics
?layers
)regularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?
-trainable_variables
?metrics
?non_trainable_variables
.	variables
?layer_metrics
?layers
/regularization_losses
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_24/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_24/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_24/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_24/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
42
53
 
?
6trainable_variables
?metrics
?non_trainable_variables
7	variables
?layer_metrics
?layers
8regularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
<trainable_variables
?metrics
?non_trainable_variables
=	variables
?layer_metrics
?layers
>regularization_losses
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_25/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_25/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_25/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_25/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
C2
D3
 
?
Etrainable_variables
?metrics
?non_trainable_variables
F	variables
?layer_metrics
?layers
Gregularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
?
Ktrainable_variables
?metrics
?non_trainable_variables
L	variables
?layer_metrics
?layers
Mregularization_losses
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_26/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_26/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_26/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_26/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

P0
Q1
R2
S3
 
?
Ttrainable_variables
?metrics
?non_trainable_variables
U	variables
?layer_metrics
?layers
Vregularization_losses
 ?layer_regularization_losses
 
 
 
?
Xtrainable_variables
?metrics
?non_trainable_variables
Y	variables
?layer_metrics
?layers
Zregularization_losses
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

\0
]1
 
?
^trainable_variables
?metrics
?non_trainable_variables
_	variables
?layer_metrics
?layers
`regularization_losses
 ?layer_regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_27/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_27/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_27/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_27/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
e2
f3
 
?
gtrainable_variables
?metrics
?non_trainable_variables
h	variables
?layer_metrics
?layers
iregularization_losses
 ?layer_regularization_losses
\Z
VARIABLE_VALUEdense_36/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_36/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

k0
l1
 
?
mtrainable_variables
?metrics
?non_trainable_variables
n	variables
?layer_metrics
?layers
oregularization_losses
 ?layer_regularization_losses
 
hf
VARIABLE_VALUEbatch_normalization_28/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_28/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_28/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_28/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

r0
s1

r0
s1
t2
u3
 
?
vtrainable_variables
?metrics
?non_trainable_variables
w	variables
?layer_metrics
?layers
xregularization_losses
 ?layer_regularization_losses
 
 
 
?
ztrainable_variables
?metrics
?non_trainable_variables
{	variables
?layer_metrics
?layers
|regularization_losses
 ?layer_regularization_losses
\Z
VARIABLE_VALUEdense_37/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_37/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1

~0
1
 
?
?trainable_variables
?metrics
?non_trainable_variables
?	variables
?layer_metrics
?layers
?regularization_losses
 ?layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
V
%0
&1
42
53
C4
D5
R6
S7
e8
f9
t10
u11
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 
 
 
 
 

%0
&1
 
 
 
 
 
 
 
 
 

40
51
 
 
 
 
 
 
 
 
 

C0
D1
 
 
 
 
 
 
 
 
 

R0
S1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

e0
f1
 
 
 
 
 
 
 
 
 

t0
u1
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputdense_31/kerneldense_31/bias&batch_normalization_23/moving_variancebatch_normalization_23/gamma"batch_normalization_23/moving_meanbatch_normalization_23/betadense_32/kerneldense_32/bias&batch_normalization_24/moving_variancebatch_normalization_24/gamma"batch_normalization_24/moving_meanbatch_normalization_24/betadense_33/kerneldense_33/bias&batch_normalization_25/moving_variancebatch_normalization_25/gamma"batch_normalization_25/moving_meanbatch_normalization_25/betadense_34/kerneldense_34/bias&batch_normalization_26/moving_variancebatch_normalization_26/gamma"batch_normalization_26/moving_meanbatch_normalization_26/betadense_35/kerneldense_35/bias&batch_normalization_27/moving_variancebatch_normalization_27/gamma"batch_normalization_27/moving_meanbatch_normalization_27/betadense_36/kerneldense_36/bias&batch_normalization_28/moving_variancebatch_normalization_28/gamma"batch_normalization_28/moving_meanbatch_normalization_28/betadense_37/kerneldense_37/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_47129614
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp0batch_normalization_24/gamma/Read/ReadVariableOp/batch_normalization_24/beta/Read/ReadVariableOp6batch_normalization_24/moving_mean/Read/ReadVariableOp:batch_normalization_24/moving_variance/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp0batch_normalization_25/gamma/Read/ReadVariableOp/batch_normalization_25/beta/Read/ReadVariableOp6batch_normalization_25/moving_mean/Read/ReadVariableOp:batch_normalization_25/moving_variance/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp0batch_normalization_26/gamma/Read/ReadVariableOp/batch_normalization_26/beta/Read/ReadVariableOp6batch_normalization_26/moving_mean/Read/ReadVariableOp:batch_normalization_26/moving_variance/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOp0batch_normalization_27/gamma/Read/ReadVariableOp/batch_normalization_27/beta/Read/ReadVariableOp6batch_normalization_27/moving_mean/Read/ReadVariableOp:batch_normalization_27/moving_variance/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp0batch_normalization_28/gamma/Read/ReadVariableOp/batch_normalization_28/beta/Read/ReadVariableOp6batch_normalization_28/moving_mean/Read/ReadVariableOp:batch_normalization_28/moving_variance/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst"/device:CPU:0*<
dtypes2
02.	
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOpAssignVariableOpdense_31/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_1AssignVariableOpdense_31/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_2AssignVariableOpbatch_normalization_23/gamma
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
k
AssignVariableOp_3AssignVariableOpbatch_normalization_23/beta
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
r
AssignVariableOp_4AssignVariableOp"batch_normalization_23/moving_mean
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
v
AssignVariableOp_5AssignVariableOp&batch_normalization_23/moving_variance
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpdense_32/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpdense_32/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_8AssignVariableOpbatch_normalization_24/gamma
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
l
AssignVariableOp_9AssignVariableOpbatch_normalization_24/betaIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_10AssignVariableOp"batch_normalization_24/moving_meanIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_11AssignVariableOp&batch_normalization_24/moving_varianceIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_12AssignVariableOpdense_33/kernelIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_13AssignVariableOpdense_33/biasIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_14AssignVariableOpbatch_normalization_25/gammaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_15AssignVariableOpbatch_normalization_25/betaIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_16AssignVariableOp"batch_normalization_25/moving_meanIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_17AssignVariableOp&batch_normalization_25/moving_varianceIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_18AssignVariableOpdense_34/kernelIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_19AssignVariableOpdense_34/biasIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_20AssignVariableOpbatch_normalization_26/gammaIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_21AssignVariableOpbatch_normalization_26/betaIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_22AssignVariableOp"batch_normalization_26/moving_meanIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_23AssignVariableOp&batch_normalization_26/moving_varianceIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_24AssignVariableOpdense_35/kernelIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_25AssignVariableOpdense_35/biasIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_26AssignVariableOpbatch_normalization_27/gammaIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_27AssignVariableOpbatch_normalization_27/betaIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_28AssignVariableOp"batch_normalization_27/moving_meanIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_29AssignVariableOp&batch_normalization_27/moving_varianceIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_30AssignVariableOpdense_36/kernelIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_31AssignVariableOpdense_36/biasIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
n
AssignVariableOp_32AssignVariableOpbatch_normalization_28/gammaIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
m
AssignVariableOp_33AssignVariableOpbatch_normalization_28/betaIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
t
AssignVariableOp_34AssignVariableOp"batch_normalization_28/moving_meanIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
x
AssignVariableOp_35AssignVariableOp&batch_normalization_28/moving_varianceIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
a
AssignVariableOp_36AssignVariableOpdense_37/kernelIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_37AssignVariableOpdense_37/biasIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0	*
_output_shapes
:
Z
AssignVariableOp_38AssignVariableOpSGD/iterIdentity_39"/device:CPU:0*
dtype0	
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_39AssignVariableOp	SGD/decayIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_40AssignVariableOpSGD/momentumIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_41AssignVariableOptotalIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_42AssignVariableOpcountIdentity_43"/device:CPU:0*
dtype0
W
Identity_44IdentityRestoreV2:43"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_43AssignVariableOptotal_1Identity_44"/device:CPU:0*
dtype0
W
Identity_45IdentityRestoreV2:44"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_44AssignVariableOpcount_1Identity_45"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_46Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??
?	
?
F__inference_dense_31_layer_call_and_return_conditional_losses_47129636

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?	
?
F__inference_dense_33_layer_call_and_return_conditional_losses_47129900

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

L
-__inference_dropout_14_layer_call_fn_47130481

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
+__inference_dense_37_layer_call_fn_47130508

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_23_layer_call_fn_47129758

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_47130396

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_47129614	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&	
 !"#$%&*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_471271192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:??????????

_user_specified_nameinput
?	
?
+__inference_dense_32_layer_call_fn_47129778

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
9__inference_batch_normalization_23_layer_call_fn_47129738

inputs
assignmovingavg_47129713
assignmovingavg_1_47129719)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129713*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47129713*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129713*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129713*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47129713AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129713*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129719*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47129719*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129719*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129719*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47129719AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129719*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
9__inference_batch_normalization_28_layer_call_fn_47130432

inputs
assignmovingavg_47130407
assignmovingavg_1_47130413)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130407*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47130407*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130407*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130407*
_output_shapes
:
2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47130407AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130407*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130413*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47130413*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130413*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130413*
_output_shapes
:
2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47130413AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130413*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
K
-__inference_dropout_14_layer_call_fn_47130486

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
+__inference_dense_36_layer_call_fn_47130340

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
+__inference_dense_33_layer_call_fn_47129910

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_24_layer_call_fn_47129890

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

L
-__inference_dropout_13_layer_call_fn_47130183

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????F2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????F*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????F2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????F2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????F2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????F:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47129682

inputs
assignmovingavg_47129657
assignmovingavg_1_47129663)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129657*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47129657*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129657*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129657*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47129657AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129657*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129663*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47129663*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129663*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129663*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47129663AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129663*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_47130376

inputs
assignmovingavg_47130351
assignmovingavg_1_47130357)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130351*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47130351*
_output_shapes
:
*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130351*
_output_shapes
:
2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130351*
_output_shapes
:
2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47130351AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130351*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130357*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47130357*
_output_shapes
:
*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130357*
_output_shapes
:
2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130357*
_output_shapes
:
2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47130357AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130357*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_26_layer_call_fn_47130154

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:F2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:F2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
K
-__inference_dropout_13_layer_call_fn_47130188

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????F2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????F2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????F:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
??
?$
E__inference_model_8_layer_call_and_return_conditional_losses_47128975	
input+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource3
/batch_normalization_23_assignmovingavg_471279715
1batch_normalization_23_assignmovingavg_1_47127977@
<batch_normalization_23_batchnorm_mul_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource3
/batch_normalization_24_assignmovingavg_471281435
1batch_normalization_24_assignmovingavg_1_47128149@
<batch_normalization_24_batchnorm_mul_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource3
/batch_normalization_25_assignmovingavg_471283155
1batch_normalization_25_assignmovingavg_1_47128321@
<batch_normalization_25_batchnorm_mul_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource3
/batch_normalization_26_assignmovingavg_471284875
1batch_normalization_26_assignmovingavg_1_47128493@
<batch_normalization_26_batchnorm_mul_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource3
/batch_normalization_27_assignmovingavg_471287035
1batch_normalization_27_assignmovingavg_1_47128709@
<batch_normalization_27_batchnorm_mul_readvariableop_resource<
8batch_normalization_27_batchnorm_readvariableop_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource3
/batch_normalization_28_assignmovingavg_471288755
1batch_normalization_28_assignmovingavg_1_47128881@
<batch_normalization_28_batchnorm_mul_readvariableop_resource<
8batch_normalization_28_batchnorm_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_23/AssignMovingAvg/ReadVariableOp?<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_23/batchnorm/ReadVariableOp?3batch_normalization_23/batchnorm/mul/ReadVariableOp?:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_24/AssignMovingAvg/ReadVariableOp?<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_24/batchnorm/ReadVariableOp?3batch_normalization_24/batchnorm/mul/ReadVariableOp?:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_25/AssignMovingAvg/ReadVariableOp?<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_25/batchnorm/ReadVariableOp?3batch_normalization_25/batchnorm/mul/ReadVariableOp?:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_26/AssignMovingAvg/ReadVariableOp?<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_26/batchnorm/ReadVariableOp?3batch_normalization_26/batchnorm/mul/ReadVariableOp?:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_27/AssignMovingAvg/ReadVariableOp?<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_27/batchnorm/ReadVariableOp?3batch_normalization_27/batchnorm/mul/ReadVariableOp?:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_28/AssignMovingAvg/ReadVariableOp?<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_28/batchnorm/ReadVariableOp?3batch_normalization_28/batchnorm/mul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOps
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_8/Const?
flatten_8/ReshapeReshapeinputflatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_8/Reshape?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMulflatten_8/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd?
5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_23/moments/mean/reduction_indices?
#batch_normalization_23/moments/meanMeandense_31/BiasAdd:output:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_23/moments/mean?
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_23/moments/StopGradient?
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferencedense_31/BiasAdd:output:04batch_normalization_23/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_23/moments/SquaredDifference?
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_23/moments/variance/reduction_indices?
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_23/moments/variance?
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_23/moments/Squeeze?
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_23/moments/Squeeze_1?
,batch_normalization_23/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47127971*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_23/AssignMovingAvg/decay?
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_23_assignmovingavg_47127971*
_output_shapes	
:?*
dtype027
5batch_normalization_23/AssignMovingAvg/ReadVariableOp?
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47127971*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/sub?
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47127971*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/mul?
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_23_assignmovingavg_47127971.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47127971*
_output_shapes
 *
dtype02<
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_23/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47127977*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_23/AssignMovingAvg_1/decay?
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_23_assignmovingavg_1_47127977*
_output_shapes	
:?*
dtype029
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47127977*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/sub?
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47127977*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/mul?
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_23_assignmovingavg_1_471279770batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47127977*
_output_shapes
 *
dtype02>
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/mul_1?
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/add_1?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMul*batch_normalization_23/batchnorm/add_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd?
5batch_normalization_24/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_24/moments/mean/reduction_indices?
#batch_normalization_24/moments/meanMeandense_32/BiasAdd:output:0>batch_normalization_24/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_24/moments/mean?
+batch_normalization_24/moments/StopGradientStopGradient,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_24/moments/StopGradient?
0batch_normalization_24/moments/SquaredDifferenceSquaredDifferencedense_32/BiasAdd:output:04batch_normalization_24/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_24/moments/SquaredDifference?
9batch_normalization_24/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_24/moments/variance/reduction_indices?
'batch_normalization_24/moments/varianceMean4batch_normalization_24/moments/SquaredDifference:z:0Bbatch_normalization_24/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_24/moments/variance?
&batch_normalization_24/moments/SqueezeSqueeze,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_24/moments/Squeeze?
(batch_normalization_24/moments/Squeeze_1Squeeze0batch_normalization_24/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_24/moments/Squeeze_1?
,batch_normalization_24/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47128143*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_24/AssignMovingAvg/decay?
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_24_assignmovingavg_47128143*
_output_shapes	
:?*
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp?
*batch_normalization_24/AssignMovingAvg/subSub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_24/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47128143*
_output_shapes	
:?2,
*batch_normalization_24/AssignMovingAvg/sub?
*batch_normalization_24/AssignMovingAvg/mulMul.batch_normalization_24/AssignMovingAvg/sub:z:05batch_normalization_24/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47128143*
_output_shapes	
:?2,
*batch_normalization_24/AssignMovingAvg/mul?
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_24_assignmovingavg_47128143.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47128143*
_output_shapes
 *
dtype02<
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_24/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47128149*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_24/AssignMovingAvg_1/decay?
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_24_assignmovingavg_1_47128149*
_output_shapes	
:?*
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_24/AssignMovingAvg_1/subSub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_24/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47128149*
_output_shapes	
:?2.
,batch_normalization_24/AssignMovingAvg_1/sub?
,batch_normalization_24/AssignMovingAvg_1/mulMul0batch_normalization_24/AssignMovingAvg_1/sub:z:07batch_normalization_24/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47128149*
_output_shapes	
:?2.
,batch_normalization_24/AssignMovingAvg_1/mul?
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_24_assignmovingavg_1_471281490batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47128149*
_output_shapes
 *
dtype02>
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_24/batchnorm/add/y?
$batch_normalization_24/batchnorm/addAddV21batch_normalization_24/moments/Squeeze_1:output:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/add?
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/Rsqrt?
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOp?
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/mul?
&batch_normalization_24/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/mul_1?
&batch_normalization_24/batchnorm/mul_2Mul/batch_normalization_24/moments/Squeeze:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/mul_2?
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_24/batchnorm/ReadVariableOp?
$batch_normalization_24/batchnorm/subSub7batch_normalization_24/batchnorm/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/sub?
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/add_1?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/BiasAdd?
5batch_normalization_25/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_25/moments/mean/reduction_indices?
#batch_normalization_25/moments/meanMeandense_33/BiasAdd:output:0>batch_normalization_25/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2%
#batch_normalization_25/moments/mean?
+batch_normalization_25/moments/StopGradientStopGradient,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes

:d2-
+batch_normalization_25/moments/StopGradient?
0batch_normalization_25/moments/SquaredDifferenceSquaredDifferencedense_33/BiasAdd:output:04batch_normalization_25/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d22
0batch_normalization_25/moments/SquaredDifference?
9batch_normalization_25/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_25/moments/variance/reduction_indices?
'batch_normalization_25/moments/varianceMean4batch_normalization_25/moments/SquaredDifference:z:0Bbatch_normalization_25/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2)
'batch_normalization_25/moments/variance?
&batch_normalization_25/moments/SqueezeSqueeze,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2(
&batch_normalization_25/moments/Squeeze?
(batch_normalization_25/moments/Squeeze_1Squeeze0batch_normalization_25/moments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2*
(batch_normalization_25/moments/Squeeze_1?
,batch_normalization_25/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47128315*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_25/AssignMovingAvg/decay?
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_25_assignmovingavg_47128315*
_output_shapes
:d*
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp?
*batch_normalization_25/AssignMovingAvg/subSub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_25/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47128315*
_output_shapes
:d2,
*batch_normalization_25/AssignMovingAvg/sub?
*batch_normalization_25/AssignMovingAvg/mulMul.batch_normalization_25/AssignMovingAvg/sub:z:05batch_normalization_25/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47128315*
_output_shapes
:d2,
*batch_normalization_25/AssignMovingAvg/mul?
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_25_assignmovingavg_47128315.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47128315*
_output_shapes
 *
dtype02<
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_25/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47128321*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_25/AssignMovingAvg_1/decay?
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_25_assignmovingavg_1_47128321*
_output_shapes
:d*
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_25/AssignMovingAvg_1/subSub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_25/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47128321*
_output_shapes
:d2.
,batch_normalization_25/AssignMovingAvg_1/sub?
,batch_normalization_25/AssignMovingAvg_1/mulMul0batch_normalization_25/AssignMovingAvg_1/sub:z:07batch_normalization_25/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47128321*
_output_shapes
:d2.
,batch_normalization_25/AssignMovingAvg_1/mul?
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_25_assignmovingavg_1_471283210batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47128321*
_output_shapes
 *
dtype02>
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_25/batchnorm/add/y?
$batch_normalization_25/batchnorm/addAddV21batch_normalization_25/moments/Squeeze_1:output:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/add?
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/Rsqrt?
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOp?
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/mul?
&batch_normalization_25/batchnorm/mul_1Muldense_33/BiasAdd:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/mul_1?
&batch_normalization_25/batchnorm/mul_2Mul/batch_normalization_25/moments/Squeeze:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/mul_2?
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype021
/batch_normalization_25/batchnorm/ReadVariableOp?
$batch_normalization_25/batchnorm/subSub7batch_normalization_25/batchnorm/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/sub?
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/add_1?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul*batch_normalization_25/batchnorm/add_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/BiasAdd?
5batch_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_26/moments/mean/reduction_indices?
#batch_normalization_26/moments/meanMeandense_34/BiasAdd:output:0>batch_normalization_26/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2%
#batch_normalization_26/moments/mean?
+batch_normalization_26/moments/StopGradientStopGradient,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes

:F2-
+batch_normalization_26/moments/StopGradient?
0batch_normalization_26/moments/SquaredDifferenceSquaredDifferencedense_34/BiasAdd:output:04batch_normalization_26/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????F22
0batch_normalization_26/moments/SquaredDifference?
9batch_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_26/moments/variance/reduction_indices?
'batch_normalization_26/moments/varianceMean4batch_normalization_26/moments/SquaredDifference:z:0Bbatch_normalization_26/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2)
'batch_normalization_26/moments/variance?
&batch_normalization_26/moments/SqueezeSqueeze,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2(
&batch_normalization_26/moments/Squeeze?
(batch_normalization_26/moments/Squeeze_1Squeeze0batch_normalization_26/moments/variance:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2*
(batch_normalization_26/moments/Squeeze_1?
,batch_normalization_26/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47128487*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_26/AssignMovingAvg/decay?
5batch_normalization_26/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_26_assignmovingavg_47128487*
_output_shapes
:F*
dtype027
5batch_normalization_26/AssignMovingAvg/ReadVariableOp?
*batch_normalization_26/AssignMovingAvg/subSub=batch_normalization_26/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_26/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47128487*
_output_shapes
:F2,
*batch_normalization_26/AssignMovingAvg/sub?
*batch_normalization_26/AssignMovingAvg/mulMul.batch_normalization_26/AssignMovingAvg/sub:z:05batch_normalization_26/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47128487*
_output_shapes
:F2,
*batch_normalization_26/AssignMovingAvg/mul?
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_26_assignmovingavg_47128487.batch_normalization_26/AssignMovingAvg/mul:z:06^batch_normalization_26/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47128487*
_output_shapes
 *
dtype02<
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_26/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47128493*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_26/AssignMovingAvg_1/decay?
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_26_assignmovingavg_1_47128493*
_output_shapes
:F*
dtype029
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_26/AssignMovingAvg_1/subSub?batch_normalization_26/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_26/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47128493*
_output_shapes
:F2.
,batch_normalization_26/AssignMovingAvg_1/sub?
,batch_normalization_26/AssignMovingAvg_1/mulMul0batch_normalization_26/AssignMovingAvg_1/sub:z:07batch_normalization_26/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47128493*
_output_shapes
:F2.
,batch_normalization_26/AssignMovingAvg_1/mul?
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_26_assignmovingavg_1_471284930batch_normalization_26/AssignMovingAvg_1/mul:z:08^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47128493*
_output_shapes
 *
dtype02>
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_26/batchnorm/add/y?
$batch_normalization_26/batchnorm/addAddV21batch_normalization_26/moments/Squeeze_1:output:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/add?
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/Rsqrt?
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOp?
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/mul?
&batch_normalization_26/batchnorm/mul_1Muldense_34/BiasAdd:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/mul_1?
&batch_normalization_26/batchnorm/mul_2Mul/batch_normalization_26/moments/Squeeze:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/mul_2?
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOp?
$batch_normalization_26/batchnorm/subSub7batch_normalization_26/batchnorm/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/sub?
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/add_1y
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_13/dropout/Const?
dropout_13/dropout/MulMul*batch_normalization_26/batchnorm/add_1:z:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:?????????F2
dropout_13/dropout/Mul?
dropout_13/dropout/ShapeShape*batch_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????F*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform?
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_13/dropout/GreaterEqual/y?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????F2!
dropout_13/dropout/GreaterEqual?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????F2
dropout_13/dropout/Cast?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????F2
dropout_13/dropout/Mul_1?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:F2*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/BiasAdd?
5batch_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_27/moments/mean/reduction_indices?
#batch_normalization_27/moments/meanMeandense_35/BiasAdd:output:0>batch_normalization_27/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2%
#batch_normalization_27/moments/mean?
+batch_normalization_27/moments/StopGradientStopGradient,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes

:22-
+batch_normalization_27/moments/StopGradient?
0batch_normalization_27/moments/SquaredDifferenceSquaredDifferencedense_35/BiasAdd:output:04batch_normalization_27/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????222
0batch_normalization_27/moments/SquaredDifference?
9batch_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_27/moments/variance/reduction_indices?
'batch_normalization_27/moments/varianceMean4batch_normalization_27/moments/SquaredDifference:z:0Bbatch_normalization_27/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2)
'batch_normalization_27/moments/variance?
&batch_normalization_27/moments/SqueezeSqueeze,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2(
&batch_normalization_27/moments/Squeeze?
(batch_normalization_27/moments/Squeeze_1Squeeze0batch_normalization_27/moments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2*
(batch_normalization_27/moments/Squeeze_1?
,batch_normalization_27/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47128703*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_27/AssignMovingAvg/decay?
5batch_normalization_27/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_27_assignmovingavg_47128703*
_output_shapes
:2*
dtype027
5batch_normalization_27/AssignMovingAvg/ReadVariableOp?
*batch_normalization_27/AssignMovingAvg/subSub=batch_normalization_27/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_27/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47128703*
_output_shapes
:22,
*batch_normalization_27/AssignMovingAvg/sub?
*batch_normalization_27/AssignMovingAvg/mulMul.batch_normalization_27/AssignMovingAvg/sub:z:05batch_normalization_27/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47128703*
_output_shapes
:22,
*batch_normalization_27/AssignMovingAvg/mul?
:batch_normalization_27/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_27_assignmovingavg_47128703.batch_normalization_27/AssignMovingAvg/mul:z:06^batch_normalization_27/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47128703*
_output_shapes
 *
dtype02<
:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_27/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47128709*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_27/AssignMovingAvg_1/decay?
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_27_assignmovingavg_1_47128709*
_output_shapes
:2*
dtype029
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_27/AssignMovingAvg_1/subSub?batch_normalization_27/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_27/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47128709*
_output_shapes
:22.
,batch_normalization_27/AssignMovingAvg_1/sub?
,batch_normalization_27/AssignMovingAvg_1/mulMul0batch_normalization_27/AssignMovingAvg_1/sub:z:07batch_normalization_27/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47128709*
_output_shapes
:22.
,batch_normalization_27/AssignMovingAvg_1/mul?
<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_27_assignmovingavg_1_471287090batch_normalization_27/AssignMovingAvg_1/mul:z:08^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47128709*
_output_shapes
 *
dtype02>
<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_27/batchnorm/add/y?
$batch_normalization_27/batchnorm/addAddV21batch_normalization_27/moments/Squeeze_1:output:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/add?
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/Rsqrt?
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype025
3batch_normalization_27/batchnorm/mul/ReadVariableOp?
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/mul?
&batch_normalization_27/batchnorm/mul_1Muldense_35/BiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/mul_1?
&batch_normalization_27/batchnorm/mul_2Mul/batch_normalization_27/moments/Squeeze:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/mul_2?
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype021
/batch_normalization_27/batchnorm/ReadVariableOp?
$batch_normalization_27/batchnorm/subSub7batch_normalization_27/batchnorm/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/sub?
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/add_1?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMul*batch_normalization_27/batchnorm/add_1:z:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/BiasAdd?
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_28/moments/mean/reduction_indices?
#batch_normalization_28/moments/meanMeandense_36/BiasAdd:output:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_28/moments/mean?
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_28/moments/StopGradient?
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferencedense_36/BiasAdd:output:04batch_normalization_28/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
22
0batch_normalization_28/moments/SquaredDifference?
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_28/moments/variance/reduction_indices?
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_28/moments/variance?
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_28/moments/Squeeze?
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_28/moments/Squeeze_1?
,batch_normalization_28/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47128875*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_28/AssignMovingAvg/decay?
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_28_assignmovingavg_47128875*
_output_shapes
:
*
dtype027
5batch_normalization_28/AssignMovingAvg/ReadVariableOp?
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47128875*
_output_shapes
:
2,
*batch_normalization_28/AssignMovingAvg/sub?
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:05batch_normalization_28/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47128875*
_output_shapes
:
2,
*batch_normalization_28/AssignMovingAvg/mul?
:batch_normalization_28/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_28_assignmovingavg_47128875.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47128875*
_output_shapes
 *
dtype02<
:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_28/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47128881*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_28/AssignMovingAvg_1/decay?
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_28_assignmovingavg_1_47128881*
_output_shapes
:
*
dtype029
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47128881*
_output_shapes
:
2.
,batch_normalization_28/AssignMovingAvg_1/sub?
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:07batch_normalization_28/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47128881*
_output_shapes
:
2.
,batch_normalization_28/AssignMovingAvg_1/mul?
<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_28_assignmovingavg_1_471288810batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47128881*
_output_shapes
 *
dtype02>
<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_28/batchnorm/add/y?
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/add?
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/Rsqrt?
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_28/batchnorm/mul/ReadVariableOp?
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/mul?
&batch_normalization_28/batchnorm/mul_1Muldense_36/BiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/mul_1?
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/mul_2?
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_28/batchnorm/ReadVariableOp?
$batch_normalization_28/batchnorm/subSub7batch_normalization_28/batchnorm/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/sub?
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/add_1y
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_14/dropout/Const?
dropout_14/dropout/MulMul*batch_normalization_28/batchnorm/add_1:z:0!dropout_14/dropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout_14/dropout/Mul?
dropout_14/dropout/ShapeShape*batch_normalization_28/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape?
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype021
/dropout_14/dropout/random_uniform/RandomUniform?
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_14/dropout/GreaterEqual/y?
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2!
dropout_14/dropout/GreaterEqual?
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout_14/dropout/Cast?
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout_14/dropout/Mul_1?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldropout_14/dropout/Mul_1:z:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:0;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_23/AssignMovingAvg/ReadVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp4^batch_normalization_23/batchnorm/mul/ReadVariableOp;^batch_normalization_24/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_24/AssignMovingAvg/ReadVariableOp=^batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_24/batchnorm/ReadVariableOp4^batch_normalization_24/batchnorm/mul/ReadVariableOp;^batch_normalization_25/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_25/AssignMovingAvg/ReadVariableOp=^batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_25/batchnorm/ReadVariableOp4^batch_normalization_25/batchnorm/mul/ReadVariableOp;^batch_normalization_26/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_26/AssignMovingAvg/ReadVariableOp=^batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_26/batchnorm/ReadVariableOp4^batch_normalization_26/batchnorm/mul/ReadVariableOp;^batch_normalization_27/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_27/AssignMovingAvg/ReadVariableOp=^batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp4^batch_normalization_27/batchnorm/mul/ReadVariableOp;^batch_normalization_28/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_28/AssignMovingAvg/ReadVariableOp=^batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp4^batch_normalization_28/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_23/AssignMovingAvg/ReadVariableOp5batch_normalization_23/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2x
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_24/AssignMovingAvg/ReadVariableOp5batch_normalization_24/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_24/batchnorm/ReadVariableOp/batch_normalization_24/batchnorm/ReadVariableOp2j
3batch_normalization_24/batchnorm/mul/ReadVariableOp3batch_normalization_24/batchnorm/mul/ReadVariableOp2x
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_25/AssignMovingAvg/ReadVariableOp5batch_normalization_25/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_25/batchnorm/ReadVariableOp/batch_normalization_25/batchnorm/ReadVariableOp2j
3batch_normalization_25/batchnorm/mul/ReadVariableOp3batch_normalization_25/batchnorm/mul/ReadVariableOp2x
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_26/AssignMovingAvg/ReadVariableOp5batch_normalization_26/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_26/batchnorm/ReadVariableOp/batch_normalization_26/batchnorm/ReadVariableOp2j
3batch_normalization_26/batchnorm/mul/ReadVariableOp3batch_normalization_26/batchnorm/mul/ReadVariableOp2x
:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_27/AssignMovingAvg/ReadVariableOp5batch_normalization_27/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2x
:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:S O
,
_output_shapes
:??????????

_user_specified_nameinput
?
?
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_47129834

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
9__inference_batch_normalization_27_layer_call_fn_47130300

inputs
assignmovingavg_47130275
assignmovingavg_1_47130281)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:22
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130275*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47130275*
_output_shapes
:2*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130275*
_output_shapes
:22
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130275*
_output_shapes
:22
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47130275AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130275*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130281*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47130281*
_output_shapes
:2*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130281*
_output_shapes
:22
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130281*
_output_shapes
:22
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47130281AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130281*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:22
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:22
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:22
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_27_layer_call_fn_47130320

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:22
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:22
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:22
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?0
?
9__inference_batch_normalization_26_layer_call_fn_47130134

inputs
assignmovingavg_47130109
assignmovingavg_1_47130115)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:F2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????F2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130109*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47130109*
_output_shapes
:F*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130109*
_output_shapes
:F2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130109*
_output_shapes
:F2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47130109AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130109*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130115*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47130115*
_output_shapes
:F*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130115*
_output_shapes
:F2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130115*
_output_shapes
:F2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47130115AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130115*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:F2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:F2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_47130166

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????F2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????F*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????F2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????F2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????F2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????F:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_47129966

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_47130244

inputs
assignmovingavg_47130219
assignmovingavg_1_47130225)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:22
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????22
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130219*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47130219*
_output_shapes
:2*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130219*
_output_shapes
:22
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130219*
_output_shapes
:22
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47130219AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130219*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130225*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47130225*
_output_shapes
:2*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130225*
_output_shapes
:22
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130225*
_output_shapes
:22
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47130225AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130225*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:22
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:22
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:22
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
??
?$
*__inference_model_8_layer_call_fn_47129380	
input+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource3
/batch_normalization_23_assignmovingavg_471291425
1batch_normalization_23_assignmovingavg_1_47129148@
<batch_normalization_23_batchnorm_mul_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource3
/batch_normalization_24_assignmovingavg_471291805
1batch_normalization_24_assignmovingavg_1_47129186@
<batch_normalization_24_batchnorm_mul_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource3
/batch_normalization_25_assignmovingavg_471292185
1batch_normalization_25_assignmovingavg_1_47129224@
<batch_normalization_25_batchnorm_mul_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource3
/batch_normalization_26_assignmovingavg_471292565
1batch_normalization_26_assignmovingavg_1_47129262@
<batch_normalization_26_batchnorm_mul_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource3
/batch_normalization_27_assignmovingavg_471293025
1batch_normalization_27_assignmovingavg_1_47129308@
<batch_normalization_27_batchnorm_mul_readvariableop_resource<
8batch_normalization_27_batchnorm_readvariableop_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource3
/batch_normalization_28_assignmovingavg_471293405
1batch_normalization_28_assignmovingavg_1_47129346@
<batch_normalization_28_batchnorm_mul_readvariableop_resource<
8batch_normalization_28_batchnorm_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_23/AssignMovingAvg/ReadVariableOp?<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_23/batchnorm/ReadVariableOp?3batch_normalization_23/batchnorm/mul/ReadVariableOp?:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_24/AssignMovingAvg/ReadVariableOp?<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_24/batchnorm/ReadVariableOp?3batch_normalization_24/batchnorm/mul/ReadVariableOp?:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_25/AssignMovingAvg/ReadVariableOp?<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_25/batchnorm/ReadVariableOp?3batch_normalization_25/batchnorm/mul/ReadVariableOp?:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_26/AssignMovingAvg/ReadVariableOp?<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_26/batchnorm/ReadVariableOp?3batch_normalization_26/batchnorm/mul/ReadVariableOp?:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_27/AssignMovingAvg/ReadVariableOp?<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_27/batchnorm/ReadVariableOp?3batch_normalization_27/batchnorm/mul/ReadVariableOp?:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp?5batch_normalization_28/AssignMovingAvg/ReadVariableOp?<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp?7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp?/batch_normalization_28/batchnorm/ReadVariableOp?3batch_normalization_28/batchnorm/mul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOps
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_8/Const?
flatten_8/ReshapeReshapeinputflatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_8/Reshape?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMulflatten_8/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd?
5batch_normalization_23/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_23/moments/mean/reduction_indices?
#batch_normalization_23/moments/meanMeandense_31/BiasAdd:output:0>batch_normalization_23/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_23/moments/mean?
+batch_normalization_23/moments/StopGradientStopGradient,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_23/moments/StopGradient?
0batch_normalization_23/moments/SquaredDifferenceSquaredDifferencedense_31/BiasAdd:output:04batch_normalization_23/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_23/moments/SquaredDifference?
9batch_normalization_23/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_23/moments/variance/reduction_indices?
'batch_normalization_23/moments/varianceMean4batch_normalization_23/moments/SquaredDifference:z:0Bbatch_normalization_23/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_23/moments/variance?
&batch_normalization_23/moments/SqueezeSqueeze,batch_normalization_23/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_23/moments/Squeeze?
(batch_normalization_23/moments/Squeeze_1Squeeze0batch_normalization_23/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_23/moments/Squeeze_1?
,batch_normalization_23/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47129142*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_23/AssignMovingAvg/decay?
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_23_assignmovingavg_47129142*
_output_shapes	
:?*
dtype027
5batch_normalization_23/AssignMovingAvg/ReadVariableOp?
*batch_normalization_23/AssignMovingAvg/subSub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_23/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47129142*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/sub?
*batch_normalization_23/AssignMovingAvg/mulMul.batch_normalization_23/AssignMovingAvg/sub:z:05batch_normalization_23/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47129142*
_output_shapes	
:?2,
*batch_normalization_23/AssignMovingAvg/mul?
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_23_assignmovingavg_47129142.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_23/AssignMovingAvg/47129142*
_output_shapes
 *
dtype02<
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_23/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47129148*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_23/AssignMovingAvg_1/decay?
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_23_assignmovingavg_1_47129148*
_output_shapes	
:?*
dtype029
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_23/AssignMovingAvg_1/subSub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_23/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47129148*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/sub?
,batch_normalization_23/AssignMovingAvg_1/mulMul0batch_normalization_23/AssignMovingAvg_1/sub:z:07batch_normalization_23/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47129148*
_output_shapes	
:?2.
,batch_normalization_23/AssignMovingAvg_1/mul?
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_23_assignmovingavg_1_471291480batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_23/AssignMovingAvg_1/47129148*
_output_shapes
 *
dtype02>
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV21batch_normalization_23/moments/Squeeze_1:output:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/mul_1?
&batch_normalization_23/batchnorm/mul_2Mul/batch_normalization_23/moments/Squeeze:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
$batch_normalization_23/batchnorm/subSub7batch_normalization_23/batchnorm/ReadVariableOp:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/add_1?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMul*batch_normalization_23/batchnorm/add_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd?
5batch_normalization_24/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_24/moments/mean/reduction_indices?
#batch_normalization_24/moments/meanMeandense_32/BiasAdd:output:0>batch_normalization_24/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2%
#batch_normalization_24/moments/mean?
+batch_normalization_24/moments/StopGradientStopGradient,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes
:	?2-
+batch_normalization_24/moments/StopGradient?
0batch_normalization_24/moments/SquaredDifferenceSquaredDifferencedense_32/BiasAdd:output:04batch_normalization_24/moments/StopGradient:output:0*
T0*(
_output_shapes
:??????????22
0batch_normalization_24/moments/SquaredDifference?
9batch_normalization_24/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_24/moments/variance/reduction_indices?
'batch_normalization_24/moments/varianceMean4batch_normalization_24/moments/SquaredDifference:z:0Bbatch_normalization_24/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2)
'batch_normalization_24/moments/variance?
&batch_normalization_24/moments/SqueezeSqueeze,batch_normalization_24/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2(
&batch_normalization_24/moments/Squeeze?
(batch_normalization_24/moments/Squeeze_1Squeeze0batch_normalization_24/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2*
(batch_normalization_24/moments/Squeeze_1?
,batch_normalization_24/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47129180*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_24/AssignMovingAvg/decay?
5batch_normalization_24/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_24_assignmovingavg_47129180*
_output_shapes	
:?*
dtype027
5batch_normalization_24/AssignMovingAvg/ReadVariableOp?
*batch_normalization_24/AssignMovingAvg/subSub=batch_normalization_24/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_24/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47129180*
_output_shapes	
:?2,
*batch_normalization_24/AssignMovingAvg/sub?
*batch_normalization_24/AssignMovingAvg/mulMul.batch_normalization_24/AssignMovingAvg/sub:z:05batch_normalization_24/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47129180*
_output_shapes	
:?2,
*batch_normalization_24/AssignMovingAvg/mul?
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_24_assignmovingavg_47129180.batch_normalization_24/AssignMovingAvg/mul:z:06^batch_normalization_24/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_24/AssignMovingAvg/47129180*
_output_shapes
 *
dtype02<
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_24/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47129186*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_24/AssignMovingAvg_1/decay?
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_24_assignmovingavg_1_47129186*
_output_shapes	
:?*
dtype029
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_24/AssignMovingAvg_1/subSub?batch_normalization_24/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_24/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47129186*
_output_shapes	
:?2.
,batch_normalization_24/AssignMovingAvg_1/sub?
,batch_normalization_24/AssignMovingAvg_1/mulMul0batch_normalization_24/AssignMovingAvg_1/sub:z:07batch_normalization_24/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47129186*
_output_shapes	
:?2.
,batch_normalization_24/AssignMovingAvg_1/mul?
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_24_assignmovingavg_1_471291860batch_normalization_24/AssignMovingAvg_1/mul:z:08^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_24/AssignMovingAvg_1/47129186*
_output_shapes
 *
dtype02>
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_24/batchnorm/add/y?
$batch_normalization_24/batchnorm/addAddV21batch_normalization_24/moments/Squeeze_1:output:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/add?
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/Rsqrt?
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOp?
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/mul?
&batch_normalization_24/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/mul_1?
&batch_normalization_24/batchnorm/mul_2Mul/batch_normalization_24/moments/Squeeze:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/mul_2?
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_24/batchnorm/ReadVariableOp?
$batch_normalization_24/batchnorm/subSub7batch_normalization_24/batchnorm/ReadVariableOp:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/sub?
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/add_1?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/BiasAdd?
5batch_normalization_25/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_25/moments/mean/reduction_indices?
#batch_normalization_25/moments/meanMeandense_33/BiasAdd:output:0>batch_normalization_25/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2%
#batch_normalization_25/moments/mean?
+batch_normalization_25/moments/StopGradientStopGradient,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes

:d2-
+batch_normalization_25/moments/StopGradient?
0batch_normalization_25/moments/SquaredDifferenceSquaredDifferencedense_33/BiasAdd:output:04batch_normalization_25/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d22
0batch_normalization_25/moments/SquaredDifference?
9batch_normalization_25/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_25/moments/variance/reduction_indices?
'batch_normalization_25/moments/varianceMean4batch_normalization_25/moments/SquaredDifference:z:0Bbatch_normalization_25/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2)
'batch_normalization_25/moments/variance?
&batch_normalization_25/moments/SqueezeSqueeze,batch_normalization_25/moments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2(
&batch_normalization_25/moments/Squeeze?
(batch_normalization_25/moments/Squeeze_1Squeeze0batch_normalization_25/moments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2*
(batch_normalization_25/moments/Squeeze_1?
,batch_normalization_25/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47129218*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_25/AssignMovingAvg/decay?
5batch_normalization_25/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_25_assignmovingavg_47129218*
_output_shapes
:d*
dtype027
5batch_normalization_25/AssignMovingAvg/ReadVariableOp?
*batch_normalization_25/AssignMovingAvg/subSub=batch_normalization_25/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_25/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47129218*
_output_shapes
:d2,
*batch_normalization_25/AssignMovingAvg/sub?
*batch_normalization_25/AssignMovingAvg/mulMul.batch_normalization_25/AssignMovingAvg/sub:z:05batch_normalization_25/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47129218*
_output_shapes
:d2,
*batch_normalization_25/AssignMovingAvg/mul?
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_25_assignmovingavg_47129218.batch_normalization_25/AssignMovingAvg/mul:z:06^batch_normalization_25/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_25/AssignMovingAvg/47129218*
_output_shapes
 *
dtype02<
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_25/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47129224*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_25/AssignMovingAvg_1/decay?
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_25_assignmovingavg_1_47129224*
_output_shapes
:d*
dtype029
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_25/AssignMovingAvg_1/subSub?batch_normalization_25/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_25/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47129224*
_output_shapes
:d2.
,batch_normalization_25/AssignMovingAvg_1/sub?
,batch_normalization_25/AssignMovingAvg_1/mulMul0batch_normalization_25/AssignMovingAvg_1/sub:z:07batch_normalization_25/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47129224*
_output_shapes
:d2.
,batch_normalization_25/AssignMovingAvg_1/mul?
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_25_assignmovingavg_1_471292240batch_normalization_25/AssignMovingAvg_1/mul:z:08^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_25/AssignMovingAvg_1/47129224*
_output_shapes
 *
dtype02>
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_25/batchnorm/add/y?
$batch_normalization_25/batchnorm/addAddV21batch_normalization_25/moments/Squeeze_1:output:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/add?
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/Rsqrt?
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOp?
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/mul?
&batch_normalization_25/batchnorm/mul_1Muldense_33/BiasAdd:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/mul_1?
&batch_normalization_25/batchnorm/mul_2Mul/batch_normalization_25/moments/Squeeze:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/mul_2?
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype021
/batch_normalization_25/batchnorm/ReadVariableOp?
$batch_normalization_25/batchnorm/subSub7batch_normalization_25/batchnorm/ReadVariableOp:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/sub?
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/add_1?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul*batch_normalization_25/batchnorm/add_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/BiasAdd?
5batch_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_26/moments/mean/reduction_indices?
#batch_normalization_26/moments/meanMeandense_34/BiasAdd:output:0>batch_normalization_26/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2%
#batch_normalization_26/moments/mean?
+batch_normalization_26/moments/StopGradientStopGradient,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes

:F2-
+batch_normalization_26/moments/StopGradient?
0batch_normalization_26/moments/SquaredDifferenceSquaredDifferencedense_34/BiasAdd:output:04batch_normalization_26/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????F22
0batch_normalization_26/moments/SquaredDifference?
9batch_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_26/moments/variance/reduction_indices?
'batch_normalization_26/moments/varianceMean4batch_normalization_26/moments/SquaredDifference:z:0Bbatch_normalization_26/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2)
'batch_normalization_26/moments/variance?
&batch_normalization_26/moments/SqueezeSqueeze,batch_normalization_26/moments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2(
&batch_normalization_26/moments/Squeeze?
(batch_normalization_26/moments/Squeeze_1Squeeze0batch_normalization_26/moments/variance:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2*
(batch_normalization_26/moments/Squeeze_1?
,batch_normalization_26/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47129256*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_26/AssignMovingAvg/decay?
5batch_normalization_26/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_26_assignmovingavg_47129256*
_output_shapes
:F*
dtype027
5batch_normalization_26/AssignMovingAvg/ReadVariableOp?
*batch_normalization_26/AssignMovingAvg/subSub=batch_normalization_26/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_26/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47129256*
_output_shapes
:F2,
*batch_normalization_26/AssignMovingAvg/sub?
*batch_normalization_26/AssignMovingAvg/mulMul.batch_normalization_26/AssignMovingAvg/sub:z:05batch_normalization_26/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47129256*
_output_shapes
:F2,
*batch_normalization_26/AssignMovingAvg/mul?
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_26_assignmovingavg_47129256.batch_normalization_26/AssignMovingAvg/mul:z:06^batch_normalization_26/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_26/AssignMovingAvg/47129256*
_output_shapes
 *
dtype02<
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_26/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47129262*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_26/AssignMovingAvg_1/decay?
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_26_assignmovingavg_1_47129262*
_output_shapes
:F*
dtype029
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_26/AssignMovingAvg_1/subSub?batch_normalization_26/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_26/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47129262*
_output_shapes
:F2.
,batch_normalization_26/AssignMovingAvg_1/sub?
,batch_normalization_26/AssignMovingAvg_1/mulMul0batch_normalization_26/AssignMovingAvg_1/sub:z:07batch_normalization_26/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47129262*
_output_shapes
:F2.
,batch_normalization_26/AssignMovingAvg_1/mul?
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_26_assignmovingavg_1_471292620batch_normalization_26/AssignMovingAvg_1/mul:z:08^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_26/AssignMovingAvg_1/47129262*
_output_shapes
 *
dtype02>
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_26/batchnorm/add/y?
$batch_normalization_26/batchnorm/addAddV21batch_normalization_26/moments/Squeeze_1:output:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/add?
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/Rsqrt?
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOp?
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/mul?
&batch_normalization_26/batchnorm/mul_1Muldense_34/BiasAdd:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/mul_1?
&batch_normalization_26/batchnorm/mul_2Mul/batch_normalization_26/moments/Squeeze:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/mul_2?
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOp?
$batch_normalization_26/batchnorm/subSub7batch_normalization_26/batchnorm/ReadVariableOp:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/sub?
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/add_1y
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_13/dropout/Const?
dropout_13/dropout/MulMul*batch_normalization_26/batchnorm/add_1:z:0!dropout_13/dropout/Const:output:0*
T0*'
_output_shapes
:?????????F2
dropout_13/dropout/Mul?
dropout_13/dropout/ShapeShape*batch_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????F*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform?
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_13/dropout/GreaterEqual/y?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????F2!
dropout_13/dropout/GreaterEqual?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????F2
dropout_13/dropout/Cast?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????F2
dropout_13/dropout/Mul_1?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:F2*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldropout_13/dropout/Mul_1:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/BiasAdd?
5batch_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_27/moments/mean/reduction_indices?
#batch_normalization_27/moments/meanMeandense_35/BiasAdd:output:0>batch_normalization_27/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2%
#batch_normalization_27/moments/mean?
+batch_normalization_27/moments/StopGradientStopGradient,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes

:22-
+batch_normalization_27/moments/StopGradient?
0batch_normalization_27/moments/SquaredDifferenceSquaredDifferencedense_35/BiasAdd:output:04batch_normalization_27/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????222
0batch_normalization_27/moments/SquaredDifference?
9batch_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_27/moments/variance/reduction_indices?
'batch_normalization_27/moments/varianceMean4batch_normalization_27/moments/SquaredDifference:z:0Bbatch_normalization_27/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:2*
	keep_dims(2)
'batch_normalization_27/moments/variance?
&batch_normalization_27/moments/SqueezeSqueeze,batch_normalization_27/moments/mean:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2(
&batch_normalization_27/moments/Squeeze?
(batch_normalization_27/moments/Squeeze_1Squeeze0batch_normalization_27/moments/variance:output:0*
T0*
_output_shapes
:2*
squeeze_dims
 2*
(batch_normalization_27/moments/Squeeze_1?
,batch_normalization_27/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47129302*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_27/AssignMovingAvg/decay?
5batch_normalization_27/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_27_assignmovingavg_47129302*
_output_shapes
:2*
dtype027
5batch_normalization_27/AssignMovingAvg/ReadVariableOp?
*batch_normalization_27/AssignMovingAvg/subSub=batch_normalization_27/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_27/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47129302*
_output_shapes
:22,
*batch_normalization_27/AssignMovingAvg/sub?
*batch_normalization_27/AssignMovingAvg/mulMul.batch_normalization_27/AssignMovingAvg/sub:z:05batch_normalization_27/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47129302*
_output_shapes
:22,
*batch_normalization_27/AssignMovingAvg/mul?
:batch_normalization_27/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_27_assignmovingavg_47129302.batch_normalization_27/AssignMovingAvg/mul:z:06^batch_normalization_27/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_27/AssignMovingAvg/47129302*
_output_shapes
 *
dtype02<
:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_27/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47129308*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_27/AssignMovingAvg_1/decay?
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_27_assignmovingavg_1_47129308*
_output_shapes
:2*
dtype029
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_27/AssignMovingAvg_1/subSub?batch_normalization_27/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_27/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47129308*
_output_shapes
:22.
,batch_normalization_27/AssignMovingAvg_1/sub?
,batch_normalization_27/AssignMovingAvg_1/mulMul0batch_normalization_27/AssignMovingAvg_1/sub:z:07batch_normalization_27/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47129308*
_output_shapes
:22.
,batch_normalization_27/AssignMovingAvg_1/mul?
<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_27_assignmovingavg_1_471293080batch_normalization_27/AssignMovingAvg_1/mul:z:08^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_27/AssignMovingAvg_1/47129308*
_output_shapes
 *
dtype02>
<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_27/batchnorm/add/y?
$batch_normalization_27/batchnorm/addAddV21batch_normalization_27/moments/Squeeze_1:output:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/add?
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/Rsqrt?
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype025
3batch_normalization_27/batchnorm/mul/ReadVariableOp?
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/mul?
&batch_normalization_27/batchnorm/mul_1Muldense_35/BiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/mul_1?
&batch_normalization_27/batchnorm/mul_2Mul/batch_normalization_27/moments/Squeeze:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/mul_2?
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype021
/batch_normalization_27/batchnorm/ReadVariableOp?
$batch_normalization_27/batchnorm/subSub7batch_normalization_27/batchnorm/ReadVariableOp:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/sub?
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/add_1?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMul*batch_normalization_27/batchnorm/add_1:z:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/BiasAdd?
5batch_normalization_28/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_28/moments/mean/reduction_indices?
#batch_normalization_28/moments/meanMeandense_36/BiasAdd:output:0>batch_normalization_28/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2%
#batch_normalization_28/moments/mean?
+batch_normalization_28/moments/StopGradientStopGradient,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes

:
2-
+batch_normalization_28/moments/StopGradient?
0batch_normalization_28/moments/SquaredDifferenceSquaredDifferencedense_36/BiasAdd:output:04batch_normalization_28/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????
22
0batch_normalization_28/moments/SquaredDifference?
9batch_normalization_28/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_28/moments/variance/reduction_indices?
'batch_normalization_28/moments/varianceMean4batch_normalization_28/moments/SquaredDifference:z:0Bbatch_normalization_28/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:
*
	keep_dims(2)
'batch_normalization_28/moments/variance?
&batch_normalization_28/moments/SqueezeSqueeze,batch_normalization_28/moments/mean:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2(
&batch_normalization_28/moments/Squeeze?
(batch_normalization_28/moments/Squeeze_1Squeeze0batch_normalization_28/moments/variance:output:0*
T0*
_output_shapes
:
*
squeeze_dims
 2*
(batch_normalization_28/moments/Squeeze_1?
,batch_normalization_28/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47129340*
_output_shapes
: *
dtype0*
valueB
 *
?#<2.
,batch_normalization_28/AssignMovingAvg/decay?
5batch_normalization_28/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_28_assignmovingavg_47129340*
_output_shapes
:
*
dtype027
5batch_normalization_28/AssignMovingAvg/ReadVariableOp?
*batch_normalization_28/AssignMovingAvg/subSub=batch_normalization_28/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_28/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47129340*
_output_shapes
:
2,
*batch_normalization_28/AssignMovingAvg/sub?
*batch_normalization_28/AssignMovingAvg/mulMul.batch_normalization_28/AssignMovingAvg/sub:z:05batch_normalization_28/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47129340*
_output_shapes
:
2,
*batch_normalization_28/AssignMovingAvg/mul?
:batch_normalization_28/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_28_assignmovingavg_47129340.batch_normalization_28/AssignMovingAvg/mul:z:06^batch_normalization_28/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*B
_class8
64loc:@batch_normalization_28/AssignMovingAvg/47129340*
_output_shapes
 *
dtype02<
:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp?
.batch_normalization_28/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47129346*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.batch_normalization_28/AssignMovingAvg_1/decay?
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_28_assignmovingavg_1_47129346*
_output_shapes
:
*
dtype029
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp?
,batch_normalization_28/AssignMovingAvg_1/subSub?batch_normalization_28/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_28/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47129346*
_output_shapes
:
2.
,batch_normalization_28/AssignMovingAvg_1/sub?
,batch_normalization_28/AssignMovingAvg_1/mulMul0batch_normalization_28/AssignMovingAvg_1/sub:z:07batch_normalization_28/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47129346*
_output_shapes
:
2.
,batch_normalization_28/AssignMovingAvg_1/mul?
<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_28_assignmovingavg_1_471293460batch_normalization_28/AssignMovingAvg_1/mul:z:08^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*D
_class:
86loc:@batch_normalization_28/AssignMovingAvg_1/47129346*
_output_shapes
 *
dtype02>
<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp?
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_28/batchnorm/add/y?
$batch_normalization_28/batchnorm/addAddV21batch_normalization_28/moments/Squeeze_1:output:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/add?
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/Rsqrt?
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_28/batchnorm/mul/ReadVariableOp?
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/mul?
&batch_normalization_28/batchnorm/mul_1Muldense_36/BiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/mul_1?
&batch_normalization_28/batchnorm/mul_2Mul/batch_normalization_28/moments/Squeeze:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/mul_2?
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_28/batchnorm/ReadVariableOp?
$batch_normalization_28/batchnorm/subSub7batch_normalization_28/batchnorm/ReadVariableOp:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/sub?
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/add_1y
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_14/dropout/Const?
dropout_14/dropout/MulMul*batch_normalization_28/batchnorm/add_1:z:0!dropout_14/dropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout_14/dropout/Mul?
dropout_14/dropout/ShapeShape*batch_normalization_28/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape?
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype021
/dropout_14/dropout/random_uniform/RandomUniform?
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_14/dropout/GreaterEqual/y?
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2!
dropout_14/dropout/GreaterEqual?
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout_14/dropout/Cast?
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout_14/dropout/Mul_1?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldropout_14/dropout/Mul_1:z:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:0;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_23/AssignMovingAvg/ReadVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_23/batchnorm/ReadVariableOp4^batch_normalization_23/batchnorm/mul/ReadVariableOp;^batch_normalization_24/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_24/AssignMovingAvg/ReadVariableOp=^batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_24/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_24/batchnorm/ReadVariableOp4^batch_normalization_24/batchnorm/mul/ReadVariableOp;^batch_normalization_25/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_25/AssignMovingAvg/ReadVariableOp=^batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_25/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_25/batchnorm/ReadVariableOp4^batch_normalization_25/batchnorm/mul/ReadVariableOp;^batch_normalization_26/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_26/AssignMovingAvg/ReadVariableOp=^batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_26/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_26/batchnorm/ReadVariableOp4^batch_normalization_26/batchnorm/mul/ReadVariableOp;^batch_normalization_27/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_27/AssignMovingAvg/ReadVariableOp=^batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_27/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp4^batch_normalization_27/batchnorm/mul/ReadVariableOp;^batch_normalization_28/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_28/AssignMovingAvg/ReadVariableOp=^batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_28/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp4^batch_normalization_28/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_23/AssignMovingAvg/ReadVariableOp5batch_normalization_23/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2x
:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp:batch_normalization_24/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_24/AssignMovingAvg/ReadVariableOp5batch_normalization_24/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_24/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp7batch_normalization_24/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_24/batchnorm/ReadVariableOp/batch_normalization_24/batchnorm/ReadVariableOp2j
3batch_normalization_24/batchnorm/mul/ReadVariableOp3batch_normalization_24/batchnorm/mul/ReadVariableOp2x
:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp:batch_normalization_25/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_25/AssignMovingAvg/ReadVariableOp5batch_normalization_25/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_25/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp7batch_normalization_25/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_25/batchnorm/ReadVariableOp/batch_normalization_25/batchnorm/ReadVariableOp2j
3batch_normalization_25/batchnorm/mul/ReadVariableOp3batch_normalization_25/batchnorm/mul/ReadVariableOp2x
:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp:batch_normalization_26/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_26/AssignMovingAvg/ReadVariableOp5batch_normalization_26/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_26/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp7batch_normalization_26/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_26/batchnorm/ReadVariableOp/batch_normalization_26/batchnorm/ReadVariableOp2j
3batch_normalization_26/batchnorm/mul/ReadVariableOp3batch_normalization_26/batchnorm/mul/ReadVariableOp2x
:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp:batch_normalization_27/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_27/AssignMovingAvg/ReadVariableOp5batch_normalization_27/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_27/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp7batch_normalization_27/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2x
:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp:batch_normalization_28/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_28/AssignMovingAvg/ReadVariableOp5batch_normalization_28/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_28/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp7batch_normalization_28/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:S O
,
_output_shapes
:??????????

_user_specified_nameinput
?
?
+__inference_dense_35_layer_call_fn_47130208

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
g
H__inference_dropout_14_layer_call_and_return_conditional_losses_47130464

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????
2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????
2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
H
,__inference_flatten_8_layer_call_fn_47129626

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_25_layer_call_fn_47130022

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_47129946

inputs
assignmovingavg_47129921
assignmovingavg_1_47129927)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129921*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47129921*
_output_shapes
:d*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129921*
_output_shapes
:d2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129921*
_output_shapes
:d2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47129921AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129921*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129927*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47129927*
_output_shapes
:d*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129927*
_output_shapes
:d2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129927*
_output_shapes
:d2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47129927AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129927*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?#
#__inference__wrapped_model_47127119	
input3
/model_8_dense_31_matmul_readvariableop_resource4
0model_8_dense_31_biasadd_readvariableop_resourceD
@model_8_batch_normalization_23_batchnorm_readvariableop_resourceH
Dmodel_8_batch_normalization_23_batchnorm_mul_readvariableop_resourceF
Bmodel_8_batch_normalization_23_batchnorm_readvariableop_1_resourceF
Bmodel_8_batch_normalization_23_batchnorm_readvariableop_2_resource3
/model_8_dense_32_matmul_readvariableop_resource4
0model_8_dense_32_biasadd_readvariableop_resourceD
@model_8_batch_normalization_24_batchnorm_readvariableop_resourceH
Dmodel_8_batch_normalization_24_batchnorm_mul_readvariableop_resourceF
Bmodel_8_batch_normalization_24_batchnorm_readvariableop_1_resourceF
Bmodel_8_batch_normalization_24_batchnorm_readvariableop_2_resource3
/model_8_dense_33_matmul_readvariableop_resource4
0model_8_dense_33_biasadd_readvariableop_resourceD
@model_8_batch_normalization_25_batchnorm_readvariableop_resourceH
Dmodel_8_batch_normalization_25_batchnorm_mul_readvariableop_resourceF
Bmodel_8_batch_normalization_25_batchnorm_readvariableop_1_resourceF
Bmodel_8_batch_normalization_25_batchnorm_readvariableop_2_resource3
/model_8_dense_34_matmul_readvariableop_resource4
0model_8_dense_34_biasadd_readvariableop_resourceD
@model_8_batch_normalization_26_batchnorm_readvariableop_resourceH
Dmodel_8_batch_normalization_26_batchnorm_mul_readvariableop_resourceF
Bmodel_8_batch_normalization_26_batchnorm_readvariableop_1_resourceF
Bmodel_8_batch_normalization_26_batchnorm_readvariableop_2_resource3
/model_8_dense_35_matmul_readvariableop_resource4
0model_8_dense_35_biasadd_readvariableop_resourceD
@model_8_batch_normalization_27_batchnorm_readvariableop_resourceH
Dmodel_8_batch_normalization_27_batchnorm_mul_readvariableop_resourceF
Bmodel_8_batch_normalization_27_batchnorm_readvariableop_1_resourceF
Bmodel_8_batch_normalization_27_batchnorm_readvariableop_2_resource3
/model_8_dense_36_matmul_readvariableop_resource4
0model_8_dense_36_biasadd_readvariableop_resourceD
@model_8_batch_normalization_28_batchnorm_readvariableop_resourceH
Dmodel_8_batch_normalization_28_batchnorm_mul_readvariableop_resourceF
Bmodel_8_batch_normalization_28_batchnorm_readvariableop_1_resourceF
Bmodel_8_batch_normalization_28_batchnorm_readvariableop_2_resource3
/model_8_dense_37_matmul_readvariableop_resource4
0model_8_dense_37_biasadd_readvariableop_resource
identity??7model_8/batch_normalization_23/batchnorm/ReadVariableOp?9model_8/batch_normalization_23/batchnorm/ReadVariableOp_1?9model_8/batch_normalization_23/batchnorm/ReadVariableOp_2?;model_8/batch_normalization_23/batchnorm/mul/ReadVariableOp?7model_8/batch_normalization_24/batchnorm/ReadVariableOp?9model_8/batch_normalization_24/batchnorm/ReadVariableOp_1?9model_8/batch_normalization_24/batchnorm/ReadVariableOp_2?;model_8/batch_normalization_24/batchnorm/mul/ReadVariableOp?7model_8/batch_normalization_25/batchnorm/ReadVariableOp?9model_8/batch_normalization_25/batchnorm/ReadVariableOp_1?9model_8/batch_normalization_25/batchnorm/ReadVariableOp_2?;model_8/batch_normalization_25/batchnorm/mul/ReadVariableOp?7model_8/batch_normalization_26/batchnorm/ReadVariableOp?9model_8/batch_normalization_26/batchnorm/ReadVariableOp_1?9model_8/batch_normalization_26/batchnorm/ReadVariableOp_2?;model_8/batch_normalization_26/batchnorm/mul/ReadVariableOp?7model_8/batch_normalization_27/batchnorm/ReadVariableOp?9model_8/batch_normalization_27/batchnorm/ReadVariableOp_1?9model_8/batch_normalization_27/batchnorm/ReadVariableOp_2?;model_8/batch_normalization_27/batchnorm/mul/ReadVariableOp?7model_8/batch_normalization_28/batchnorm/ReadVariableOp?9model_8/batch_normalization_28/batchnorm/ReadVariableOp_1?9model_8/batch_normalization_28/batchnorm/ReadVariableOp_2?;model_8/batch_normalization_28/batchnorm/mul/ReadVariableOp?'model_8/dense_31/BiasAdd/ReadVariableOp?&model_8/dense_31/MatMul/ReadVariableOp?'model_8/dense_32/BiasAdd/ReadVariableOp?&model_8/dense_32/MatMul/ReadVariableOp?'model_8/dense_33/BiasAdd/ReadVariableOp?&model_8/dense_33/MatMul/ReadVariableOp?'model_8/dense_34/BiasAdd/ReadVariableOp?&model_8/dense_34/MatMul/ReadVariableOp?'model_8/dense_35/BiasAdd/ReadVariableOp?&model_8/dense_35/MatMul/ReadVariableOp?'model_8/dense_36/BiasAdd/ReadVariableOp?&model_8/dense_36/MatMul/ReadVariableOp?'model_8/dense_37/BiasAdd/ReadVariableOp?&model_8/dense_37/MatMul/ReadVariableOp?
model_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model_8/flatten_8/Const?
model_8/flatten_8/ReshapeReshapeinput model_8/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
model_8/flatten_8/Reshape?
&model_8/dense_31/MatMul/ReadVariableOpReadVariableOp/model_8_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02(
&model_8/dense_31/MatMul/ReadVariableOp?
model_8/dense_31/MatMulMatMul"model_8/flatten_8/Reshape:output:0.model_8/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_31/MatMul?
'model_8/dense_31/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_8/dense_31/BiasAdd/ReadVariableOp?
model_8/dense_31/BiasAddBiasAdd!model_8/dense_31/MatMul:product:0/model_8/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_31/BiasAdd?
7model_8/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_8/batch_normalization_23/batchnorm/ReadVariableOp?
.model_8/batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_8/batch_normalization_23/batchnorm/add/y?
,model_8/batch_normalization_23/batchnorm/addAddV2?model_8/batch_normalization_23/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,model_8/batch_normalization_23/batchnorm/add?
.model_8/batch_normalization_23/batchnorm/RsqrtRsqrt0model_8/batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.model_8/batch_normalization_23/batchnorm/Rsqrt?
;model_8/batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model_8/batch_normalization_23/batchnorm/mul/ReadVariableOp?
,model_8/batch_normalization_23/batchnorm/mulMul2model_8/batch_normalization_23/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,model_8/batch_normalization_23/batchnorm/mul?
.model_8/batch_normalization_23/batchnorm/mul_1Mul!model_8/dense_31/BiasAdd:output:00model_8/batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.model_8/batch_normalization_23/batchnorm/mul_1?
9model_8/batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9model_8/batch_normalization_23/batchnorm/ReadVariableOp_1?
.model_8/batch_normalization_23/batchnorm/mul_2MulAmodel_8/batch_normalization_23/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.model_8/batch_normalization_23/batchnorm/mul_2?
9model_8/batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9model_8/batch_normalization_23/batchnorm/ReadVariableOp_2?
,model_8/batch_normalization_23/batchnorm/subSubAmodel_8/batch_normalization_23/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,model_8/batch_normalization_23/batchnorm/sub?
.model_8/batch_normalization_23/batchnorm/add_1AddV22model_8/batch_normalization_23/batchnorm/mul_1:z:00model_8/batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.model_8/batch_normalization_23/batchnorm/add_1?
&model_8/dense_32/MatMul/ReadVariableOpReadVariableOp/model_8_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_8/dense_32/MatMul/ReadVariableOp?
model_8/dense_32/MatMulMatMul2model_8/batch_normalization_23/batchnorm/add_1:z:0.model_8/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_32/MatMul?
'model_8/dense_32/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_8/dense_32/BiasAdd/ReadVariableOp?
model_8/dense_32/BiasAddBiasAdd!model_8/dense_32/MatMul:product:0/model_8/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_8/dense_32/BiasAdd?
7model_8/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_8/batch_normalization_24/batchnorm/ReadVariableOp?
.model_8/batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_8/batch_normalization_24/batchnorm/add/y?
,model_8/batch_normalization_24/batchnorm/addAddV2?model_8/batch_normalization_24/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,model_8/batch_normalization_24/batchnorm/add?
.model_8/batch_normalization_24/batchnorm/RsqrtRsqrt0model_8/batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.model_8/batch_normalization_24/batchnorm/Rsqrt?
;model_8/batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;model_8/batch_normalization_24/batchnorm/mul/ReadVariableOp?
,model_8/batch_normalization_24/batchnorm/mulMul2model_8/batch_normalization_24/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,model_8/batch_normalization_24/batchnorm/mul?
.model_8/batch_normalization_24/batchnorm/mul_1Mul!model_8/dense_32/BiasAdd:output:00model_8/batch_normalization_24/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????20
.model_8/batch_normalization_24/batchnorm/mul_1?
9model_8/batch_normalization_24/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_24_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9model_8/batch_normalization_24/batchnorm/ReadVariableOp_1?
.model_8/batch_normalization_24/batchnorm/mul_2MulAmodel_8/batch_normalization_24/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.model_8/batch_normalization_24/batchnorm/mul_2?
9model_8/batch_normalization_24/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_24_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9model_8/batch_normalization_24/batchnorm/ReadVariableOp_2?
,model_8/batch_normalization_24/batchnorm/subSubAmodel_8/batch_normalization_24/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,model_8/batch_normalization_24/batchnorm/sub?
.model_8/batch_normalization_24/batchnorm/add_1AddV22model_8/batch_normalization_24/batchnorm/mul_1:z:00model_8/batch_normalization_24/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????20
.model_8/batch_normalization_24/batchnorm/add_1?
&model_8/dense_33/MatMul/ReadVariableOpReadVariableOp/model_8_dense_33_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02(
&model_8/dense_33/MatMul/ReadVariableOp?
model_8/dense_33/MatMulMatMul2model_8/batch_normalization_24/batchnorm/add_1:z:0.model_8/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_8/dense_33/MatMul?
'model_8/dense_33/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_33_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'model_8/dense_33/BiasAdd/ReadVariableOp?
model_8/dense_33/BiasAddBiasAdd!model_8/dense_33/MatMul:product:0/model_8/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
model_8/dense_33/BiasAdd?
7model_8/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype029
7model_8/batch_normalization_25/batchnorm/ReadVariableOp?
.model_8/batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_8/batch_normalization_25/batchnorm/add/y?
,model_8/batch_normalization_25/batchnorm/addAddV2?model_8/batch_normalization_25/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2.
,model_8/batch_normalization_25/batchnorm/add?
.model_8/batch_normalization_25/batchnorm/RsqrtRsqrt0model_8/batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:d20
.model_8/batch_normalization_25/batchnorm/Rsqrt?
;model_8/batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02=
;model_8/batch_normalization_25/batchnorm/mul/ReadVariableOp?
,model_8/batch_normalization_25/batchnorm/mulMul2model_8/batch_normalization_25/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2.
,model_8/batch_normalization_25/batchnorm/mul?
.model_8/batch_normalization_25/batchnorm/mul_1Mul!model_8/dense_33/BiasAdd:output:00model_8/batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d20
.model_8/batch_normalization_25/batchnorm/mul_1?
9model_8/batch_normalization_25/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_25_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype02;
9model_8/batch_normalization_25/batchnorm/ReadVariableOp_1?
.model_8/batch_normalization_25/batchnorm/mul_2MulAmodel_8/batch_normalization_25/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:d20
.model_8/batch_normalization_25/batchnorm/mul_2?
9model_8/batch_normalization_25/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_25_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype02;
9model_8/batch_normalization_25/batchnorm/ReadVariableOp_2?
,model_8/batch_normalization_25/batchnorm/subSubAmodel_8/batch_normalization_25/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2.
,model_8/batch_normalization_25/batchnorm/sub?
.model_8/batch_normalization_25/batchnorm/add_1AddV22model_8/batch_normalization_25/batchnorm/mul_1:z:00model_8/batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d20
.model_8/batch_normalization_25/batchnorm/add_1?
&model_8/dense_34/MatMul/ReadVariableOpReadVariableOp/model_8_dense_34_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02(
&model_8/dense_34/MatMul/ReadVariableOp?
model_8/dense_34/MatMulMatMul2model_8/batch_normalization_25/batchnorm/add_1:z:0.model_8/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_8/dense_34/MatMul?
'model_8/dense_34/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_34_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02)
'model_8/dense_34/BiasAdd/ReadVariableOp?
model_8/dense_34/BiasAddBiasAdd!model_8/dense_34/MatMul:product:0/model_8/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
model_8/dense_34/BiasAdd?
7model_8/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype029
7model_8/batch_normalization_26/batchnorm/ReadVariableOp?
.model_8/batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_8/batch_normalization_26/batchnorm/add/y?
,model_8/batch_normalization_26/batchnorm/addAddV2?model_8/batch_normalization_26/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:F2.
,model_8/batch_normalization_26/batchnorm/add?
.model_8/batch_normalization_26/batchnorm/RsqrtRsqrt0model_8/batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:F20
.model_8/batch_normalization_26/batchnorm/Rsqrt?
;model_8/batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype02=
;model_8/batch_normalization_26/batchnorm/mul/ReadVariableOp?
,model_8/batch_normalization_26/batchnorm/mulMul2model_8/batch_normalization_26/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2.
,model_8/batch_normalization_26/batchnorm/mul?
.model_8/batch_normalization_26/batchnorm/mul_1Mul!model_8/dense_34/BiasAdd:output:00model_8/batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F20
.model_8/batch_normalization_26/batchnorm/mul_1?
9model_8/batch_normalization_26/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_26_batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype02;
9model_8/batch_normalization_26/batchnorm/ReadVariableOp_1?
.model_8/batch_normalization_26/batchnorm/mul_2MulAmodel_8/batch_normalization_26/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:F20
.model_8/batch_normalization_26/batchnorm/mul_2?
9model_8/batch_normalization_26/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_26_batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype02;
9model_8/batch_normalization_26/batchnorm/ReadVariableOp_2?
,model_8/batch_normalization_26/batchnorm/subSubAmodel_8/batch_normalization_26/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2.
,model_8/batch_normalization_26/batchnorm/sub?
.model_8/batch_normalization_26/batchnorm/add_1AddV22model_8/batch_normalization_26/batchnorm/mul_1:z:00model_8/batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F20
.model_8/batch_normalization_26/batchnorm/add_1?
model_8/dropout_13/IdentityIdentity2model_8/batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????F2
model_8/dropout_13/Identity?
&model_8/dense_35/MatMul/ReadVariableOpReadVariableOp/model_8_dense_35_matmul_readvariableop_resource*
_output_shapes

:F2*
dtype02(
&model_8/dense_35/MatMul/ReadVariableOp?
model_8/dense_35/MatMulMatMul$model_8/dropout_13/Identity:output:0.model_8/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_8/dense_35/MatMul?
'model_8/dense_35/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02)
'model_8/dense_35/BiasAdd/ReadVariableOp?
model_8/dense_35/BiasAddBiasAdd!model_8/dense_35/MatMul:product:0/model_8/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
model_8/dense_35/BiasAdd?
7model_8/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype029
7model_8/batch_normalization_27/batchnorm/ReadVariableOp?
.model_8/batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_8/batch_normalization_27/batchnorm/add/y?
,model_8/batch_normalization_27/batchnorm/addAddV2?model_8/batch_normalization_27/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:22.
,model_8/batch_normalization_27/batchnorm/add?
.model_8/batch_normalization_27/batchnorm/RsqrtRsqrt0model_8/batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:220
.model_8/batch_normalization_27/batchnorm/Rsqrt?
;model_8/batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype02=
;model_8/batch_normalization_27/batchnorm/mul/ReadVariableOp?
,model_8/batch_normalization_27/batchnorm/mulMul2model_8/batch_normalization_27/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22.
,model_8/batch_normalization_27/batchnorm/mul?
.model_8/batch_normalization_27/batchnorm/mul_1Mul!model_8/dense_35/BiasAdd:output:00model_8/batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????220
.model_8/batch_normalization_27/batchnorm/mul_1?
9model_8/batch_normalization_27/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype02;
9model_8/batch_normalization_27/batchnorm/ReadVariableOp_1?
.model_8/batch_normalization_27/batchnorm/mul_2MulAmodel_8/batch_normalization_27/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:220
.model_8/batch_normalization_27/batchnorm/mul_2?
9model_8/batch_normalization_27/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype02;
9model_8/batch_normalization_27/batchnorm/ReadVariableOp_2?
,model_8/batch_normalization_27/batchnorm/subSubAmodel_8/batch_normalization_27/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22.
,model_8/batch_normalization_27/batchnorm/sub?
.model_8/batch_normalization_27/batchnorm/add_1AddV22model_8/batch_normalization_27/batchnorm/mul_1:z:00model_8/batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????220
.model_8/batch_normalization_27/batchnorm/add_1?
&model_8/dense_36/MatMul/ReadVariableOpReadVariableOp/model_8_dense_36_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02(
&model_8/dense_36/MatMul/ReadVariableOp?
model_8/dense_36/MatMulMatMul2model_8/batch_normalization_27/batchnorm/add_1:z:0.model_8/dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_8/dense_36/MatMul?
'model_8/dense_36/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_36_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_8/dense_36/BiasAdd/ReadVariableOp?
model_8/dense_36/BiasAddBiasAdd!model_8/dense_36/MatMul:product:0/model_8/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model_8/dense_36/BiasAdd?
7model_8/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp@model_8_batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype029
7model_8/batch_normalization_28/batchnorm/ReadVariableOp?
.model_8/batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.model_8/batch_normalization_28/batchnorm/add/y?
,model_8/batch_normalization_28/batchnorm/addAddV2?model_8/batch_normalization_28/batchnorm/ReadVariableOp:value:07model_8/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2.
,model_8/batch_normalization_28/batchnorm/add?
.model_8/batch_normalization_28/batchnorm/RsqrtRsqrt0model_8/batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:
20
.model_8/batch_normalization_28/batchnorm/Rsqrt?
;model_8/batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOpDmodel_8_batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02=
;model_8/batch_normalization_28/batchnorm/mul/ReadVariableOp?
,model_8/batch_normalization_28/batchnorm/mulMul2model_8/batch_normalization_28/batchnorm/Rsqrt:y:0Cmodel_8/batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2.
,model_8/batch_normalization_28/batchnorm/mul?
.model_8/batch_normalization_28/batchnorm/mul_1Mul!model_8/dense_36/BiasAdd:output:00model_8/batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
20
.model_8/batch_normalization_28/batchnorm/mul_1?
9model_8/batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOpBmodel_8_batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02;
9model_8/batch_normalization_28/batchnorm/ReadVariableOp_1?
.model_8/batch_normalization_28/batchnorm/mul_2MulAmodel_8/batch_normalization_28/batchnorm/ReadVariableOp_1:value:00model_8/batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:
20
.model_8/batch_normalization_28/batchnorm/mul_2?
9model_8/batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOpBmodel_8_batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02;
9model_8/batch_normalization_28/batchnorm/ReadVariableOp_2?
,model_8/batch_normalization_28/batchnorm/subSubAmodel_8/batch_normalization_28/batchnorm/ReadVariableOp_2:value:02model_8/batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2.
,model_8/batch_normalization_28/batchnorm/sub?
.model_8/batch_normalization_28/batchnorm/add_1AddV22model_8/batch_normalization_28/batchnorm/mul_1:z:00model_8/batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
20
.model_8/batch_normalization_28/batchnorm/add_1?
model_8/dropout_14/IdentityIdentity2model_8/batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????
2
model_8/dropout_14/Identity?
&model_8/dense_37/MatMul/ReadVariableOpReadVariableOp/model_8_dense_37_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02(
&model_8/dense_37/MatMul/ReadVariableOp?
model_8/dense_37/MatMulMatMul$model_8/dropout_14/Identity:output:0.model_8/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/dense_37/MatMul?
'model_8/dense_37/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_8/dense_37/BiasAdd/ReadVariableOp?
model_8/dense_37/BiasAddBiasAdd!model_8/dense_37/MatMul:product:0/model_8/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/dense_37/BiasAdd?
model_8/dense_37/SoftmaxSoftmax!model_8/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_8/dense_37/Softmax?
IdentityIdentity"model_8/dense_37/Softmax:softmax:08^model_8/batch_normalization_23/batchnorm/ReadVariableOp:^model_8/batch_normalization_23/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_23/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_23/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_24/batchnorm/ReadVariableOp:^model_8/batch_normalization_24/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_24/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_24/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_25/batchnorm/ReadVariableOp:^model_8/batch_normalization_25/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_25/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_25/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_26/batchnorm/ReadVariableOp:^model_8/batch_normalization_26/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_26/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_26/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_27/batchnorm/ReadVariableOp:^model_8/batch_normalization_27/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_27/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_27/batchnorm/mul/ReadVariableOp8^model_8/batch_normalization_28/batchnorm/ReadVariableOp:^model_8/batch_normalization_28/batchnorm/ReadVariableOp_1:^model_8/batch_normalization_28/batchnorm/ReadVariableOp_2<^model_8/batch_normalization_28/batchnorm/mul/ReadVariableOp(^model_8/dense_31/BiasAdd/ReadVariableOp'^model_8/dense_31/MatMul/ReadVariableOp(^model_8/dense_32/BiasAdd/ReadVariableOp'^model_8/dense_32/MatMul/ReadVariableOp(^model_8/dense_33/BiasAdd/ReadVariableOp'^model_8/dense_33/MatMul/ReadVariableOp(^model_8/dense_34/BiasAdd/ReadVariableOp'^model_8/dense_34/MatMul/ReadVariableOp(^model_8/dense_35/BiasAdd/ReadVariableOp'^model_8/dense_35/MatMul/ReadVariableOp(^model_8/dense_36/BiasAdd/ReadVariableOp'^model_8/dense_36/MatMul/ReadVariableOp(^model_8/dense_37/BiasAdd/ReadVariableOp'^model_8/dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::2r
7model_8/batch_normalization_23/batchnorm/ReadVariableOp7model_8/batch_normalization_23/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_23/batchnorm/ReadVariableOp_19model_8/batch_normalization_23/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_23/batchnorm/ReadVariableOp_29model_8/batch_normalization_23/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_23/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_23/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_24/batchnorm/ReadVariableOp7model_8/batch_normalization_24/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_24/batchnorm/ReadVariableOp_19model_8/batch_normalization_24/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_24/batchnorm/ReadVariableOp_29model_8/batch_normalization_24/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_24/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_24/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_25/batchnorm/ReadVariableOp7model_8/batch_normalization_25/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_25/batchnorm/ReadVariableOp_19model_8/batch_normalization_25/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_25/batchnorm/ReadVariableOp_29model_8/batch_normalization_25/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_25/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_25/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_26/batchnorm/ReadVariableOp7model_8/batch_normalization_26/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_26/batchnorm/ReadVariableOp_19model_8/batch_normalization_26/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_26/batchnorm/ReadVariableOp_29model_8/batch_normalization_26/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_26/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_26/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_27/batchnorm/ReadVariableOp7model_8/batch_normalization_27/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_27/batchnorm/ReadVariableOp_19model_8/batch_normalization_27/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_27/batchnorm/ReadVariableOp_29model_8/batch_normalization_27/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_27/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_27/batchnorm/mul/ReadVariableOp2r
7model_8/batch_normalization_28/batchnorm/ReadVariableOp7model_8/batch_normalization_28/batchnorm/ReadVariableOp2v
9model_8/batch_normalization_28/batchnorm/ReadVariableOp_19model_8/batch_normalization_28/batchnorm/ReadVariableOp_12v
9model_8/batch_normalization_28/batchnorm/ReadVariableOp_29model_8/batch_normalization_28/batchnorm/ReadVariableOp_22z
;model_8/batch_normalization_28/batchnorm/mul/ReadVariableOp;model_8/batch_normalization_28/batchnorm/mul/ReadVariableOp2R
'model_8/dense_31/BiasAdd/ReadVariableOp'model_8/dense_31/BiasAdd/ReadVariableOp2P
&model_8/dense_31/MatMul/ReadVariableOp&model_8/dense_31/MatMul/ReadVariableOp2R
'model_8/dense_32/BiasAdd/ReadVariableOp'model_8/dense_32/BiasAdd/ReadVariableOp2P
&model_8/dense_32/MatMul/ReadVariableOp&model_8/dense_32/MatMul/ReadVariableOp2R
'model_8/dense_33/BiasAdd/ReadVariableOp'model_8/dense_33/BiasAdd/ReadVariableOp2P
&model_8/dense_33/MatMul/ReadVariableOp&model_8/dense_33/MatMul/ReadVariableOp2R
'model_8/dense_34/BiasAdd/ReadVariableOp'model_8/dense_34/BiasAdd/ReadVariableOp2P
&model_8/dense_34/MatMul/ReadVariableOp&model_8/dense_34/MatMul/ReadVariableOp2R
'model_8/dense_35/BiasAdd/ReadVariableOp'model_8/dense_35/BiasAdd/ReadVariableOp2P
&model_8/dense_35/MatMul/ReadVariableOp&model_8/dense_35/MatMul/ReadVariableOp2R
'model_8/dense_36/BiasAdd/ReadVariableOp'model_8/dense_36/BiasAdd/ReadVariableOp2P
&model_8/dense_36/MatMul/ReadVariableOp&model_8/dense_36/MatMul/ReadVariableOp2R
'model_8/dense_37/BiasAdd/ReadVariableOp'model_8/dense_37/BiasAdd/ReadVariableOp2P
&model_8/dense_37/MatMul/ReadVariableOp&model_8/dense_37/MatMul/ReadVariableOp:S O
,
_output_shapes
:??????????

_user_specified_nameinput
?0
?
9__inference_batch_normalization_24_layer_call_fn_47129870

inputs
assignmovingavg_47129845
assignmovingavg_1_47129851)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129845*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47129845*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129845*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129845*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47129845AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129845*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129851*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47129851*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129851*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129851*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47129851AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129851*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47129702

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_34_layer_call_and_return_conditional_losses_47130032

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
F__inference_dense_36_layer_call_and_return_conditional_losses_47130330

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?	
?
+__inference_dense_31_layer_call_fn_47129646

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_47130078

inputs
assignmovingavg_47130053
assignmovingavg_1_47130059)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:F2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????F2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:F*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:F*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130053*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47130053*
_output_shapes
:F*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130053*
_output_shapes
:F2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47130053*
_output_shapes
:F2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47130053AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47130053*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130059*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47130059*
_output_shapes
:F*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130059*
_output_shapes
:F2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47130059*
_output_shapes
:F2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47130059AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47130059*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:F2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:F2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
??
?
*__inference_model_8_layer_call_fn_47129527	
input+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource@
<batch_normalization_23_batchnorm_mul_readvariableop_resource>
:batch_normalization_23_batchnorm_readvariableop_1_resource>
:batch_normalization_23_batchnorm_readvariableop_2_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource@
<batch_normalization_24_batchnorm_mul_readvariableop_resource>
:batch_normalization_24_batchnorm_readvariableop_1_resource>
:batch_normalization_24_batchnorm_readvariableop_2_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource@
<batch_normalization_25_batchnorm_mul_readvariableop_resource>
:batch_normalization_25_batchnorm_readvariableop_1_resource>
:batch_normalization_25_batchnorm_readvariableop_2_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource@
<batch_normalization_26_batchnorm_mul_readvariableop_resource>
:batch_normalization_26_batchnorm_readvariableop_1_resource>
:batch_normalization_26_batchnorm_readvariableop_2_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource<
8batch_normalization_27_batchnorm_readvariableop_resource@
<batch_normalization_27_batchnorm_mul_readvariableop_resource>
:batch_normalization_27_batchnorm_readvariableop_1_resource>
:batch_normalization_27_batchnorm_readvariableop_2_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource<
8batch_normalization_28_batchnorm_readvariableop_resource@
<batch_normalization_28_batchnorm_mul_readvariableop_resource>
:batch_normalization_28_batchnorm_readvariableop_1_resource>
:batch_normalization_28_batchnorm_readvariableop_2_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??/batch_normalization_23/batchnorm/ReadVariableOp?1batch_normalization_23/batchnorm/ReadVariableOp_1?1batch_normalization_23/batchnorm/ReadVariableOp_2?3batch_normalization_23/batchnorm/mul/ReadVariableOp?/batch_normalization_24/batchnorm/ReadVariableOp?1batch_normalization_24/batchnorm/ReadVariableOp_1?1batch_normalization_24/batchnorm/ReadVariableOp_2?3batch_normalization_24/batchnorm/mul/ReadVariableOp?/batch_normalization_25/batchnorm/ReadVariableOp?1batch_normalization_25/batchnorm/ReadVariableOp_1?1batch_normalization_25/batchnorm/ReadVariableOp_2?3batch_normalization_25/batchnorm/mul/ReadVariableOp?/batch_normalization_26/batchnorm/ReadVariableOp?1batch_normalization_26/batchnorm/ReadVariableOp_1?1batch_normalization_26/batchnorm/ReadVariableOp_2?3batch_normalization_26/batchnorm/mul/ReadVariableOp?/batch_normalization_27/batchnorm/ReadVariableOp?1batch_normalization_27/batchnorm/ReadVariableOp_1?1batch_normalization_27/batchnorm/ReadVariableOp_2?3batch_normalization_27/batchnorm/mul/ReadVariableOp?/batch_normalization_28/batchnorm/ReadVariableOp?1batch_normalization_28/batchnorm/ReadVariableOp_1?1batch_normalization_28/batchnorm/ReadVariableOp_2?3batch_normalization_28/batchnorm/mul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOps
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_8/Const?
flatten_8/ReshapeReshapeinputflatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_8/Reshape?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMulflatten_8/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/mul_1?
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_1?
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_2?
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/add_1?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMul*batch_normalization_23/batchnorm/add_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd?
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_24/batchnorm/ReadVariableOp?
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_24/batchnorm/add/y?
$batch_normalization_24/batchnorm/addAddV27batch_normalization_24/batchnorm/ReadVariableOp:value:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/add?
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/Rsqrt?
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOp?
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/mul?
&batch_normalization_24/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/mul_1?
1batch_normalization_24/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_1?
&batch_normalization_24/batchnorm/mul_2Mul9batch_normalization_24/batchnorm/ReadVariableOp_1:value:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/mul_2?
1batch_normalization_24/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_2?
$batch_normalization_24/batchnorm/subSub9batch_normalization_24/batchnorm/ReadVariableOp_2:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/sub?
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/add_1?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/BiasAdd?
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype021
/batch_normalization_25/batchnorm/ReadVariableOp?
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_25/batchnorm/add/y?
$batch_normalization_25/batchnorm/addAddV27batch_normalization_25/batchnorm/ReadVariableOp:value:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/add?
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/Rsqrt?
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOp?
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/mul?
&batch_normalization_25/batchnorm/mul_1Muldense_33/BiasAdd:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/mul_1?
1batch_normalization_25/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_1?
&batch_normalization_25/batchnorm/mul_2Mul9batch_normalization_25/batchnorm/ReadVariableOp_1:value:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/mul_2?
1batch_normalization_25/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_2?
$batch_normalization_25/batchnorm/subSub9batch_normalization_25/batchnorm/ReadVariableOp_2:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/sub?
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/add_1?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul*batch_normalization_25/batchnorm/add_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/BiasAdd?
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOp?
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_26/batchnorm/add/y?
$batch_normalization_26/batchnorm/addAddV27batch_normalization_26/batchnorm/ReadVariableOp:value:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/add?
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/Rsqrt?
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOp?
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/mul?
&batch_normalization_26/batchnorm/mul_1Muldense_34/BiasAdd:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/mul_1?
1batch_normalization_26/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_1?
&batch_normalization_26/batchnorm/mul_2Mul9batch_normalization_26/batchnorm/ReadVariableOp_1:value:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/mul_2?
1batch_normalization_26/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_2?
$batch_normalization_26/batchnorm/subSub9batch_normalization_26/batchnorm/ReadVariableOp_2:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/sub?
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/add_1?
dropout_13/IdentityIdentity*batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????F2
dropout_13/Identity?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:F2*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldropout_13/Identity:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/BiasAdd?
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype021
/batch_normalization_27/batchnorm/ReadVariableOp?
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_27/batchnorm/add/y?
$batch_normalization_27/batchnorm/addAddV27batch_normalization_27/batchnorm/ReadVariableOp:value:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/add?
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/Rsqrt?
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype025
3batch_normalization_27/batchnorm/mul/ReadVariableOp?
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/mul?
&batch_normalization_27/batchnorm/mul_1Muldense_35/BiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/mul_1?
1batch_normalization_27/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype023
1batch_normalization_27/batchnorm/ReadVariableOp_1?
&batch_normalization_27/batchnorm/mul_2Mul9batch_normalization_27/batchnorm/ReadVariableOp_1:value:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/mul_2?
1batch_normalization_27/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype023
1batch_normalization_27/batchnorm/ReadVariableOp_2?
$batch_normalization_27/batchnorm/subSub9batch_normalization_27/batchnorm/ReadVariableOp_2:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/sub?
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/add_1?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMul*batch_normalization_27/batchnorm/add_1:z:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/BiasAdd?
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_28/batchnorm/ReadVariableOp?
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_28/batchnorm/add/y?
$batch_normalization_28/batchnorm/addAddV27batch_normalization_28/batchnorm/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/add?
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/Rsqrt?
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_28/batchnorm/mul/ReadVariableOp?
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/mul?
&batch_normalization_28/batchnorm/mul_1Muldense_36/BiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/mul_1?
1batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_28/batchnorm/ReadVariableOp_1?
&batch_normalization_28/batchnorm/mul_2Mul9batch_normalization_28/batchnorm/ReadVariableOp_1:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/mul_2?
1batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_28/batchnorm/ReadVariableOp_2?
$batch_normalization_28/batchnorm/subSub9batch_normalization_28/batchnorm/ReadVariableOp_2:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/sub?
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/add_1?
dropout_14/IdentityIdentity*batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????
2
dropout_14/Identity?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldropout_14/Identity:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:00^batch_normalization_23/batchnorm/ReadVariableOp2^batch_normalization_23/batchnorm/ReadVariableOp_12^batch_normalization_23/batchnorm/ReadVariableOp_24^batch_normalization_23/batchnorm/mul/ReadVariableOp0^batch_normalization_24/batchnorm/ReadVariableOp2^batch_normalization_24/batchnorm/ReadVariableOp_12^batch_normalization_24/batchnorm/ReadVariableOp_24^batch_normalization_24/batchnorm/mul/ReadVariableOp0^batch_normalization_25/batchnorm/ReadVariableOp2^batch_normalization_25/batchnorm/ReadVariableOp_12^batch_normalization_25/batchnorm/ReadVariableOp_24^batch_normalization_25/batchnorm/mul/ReadVariableOp0^batch_normalization_26/batchnorm/ReadVariableOp2^batch_normalization_26/batchnorm/ReadVariableOp_12^batch_normalization_26/batchnorm/ReadVariableOp_24^batch_normalization_26/batchnorm/mul/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp2^batch_normalization_27/batchnorm/ReadVariableOp_12^batch_normalization_27/batchnorm/ReadVariableOp_24^batch_normalization_27/batchnorm/mul/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp2^batch_normalization_28/batchnorm/ReadVariableOp_12^batch_normalization_28/batchnorm/ReadVariableOp_24^batch_normalization_28/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2f
1batch_normalization_23/batchnorm/ReadVariableOp_11batch_normalization_23/batchnorm/ReadVariableOp_12f
1batch_normalization_23/batchnorm/ReadVariableOp_21batch_normalization_23/batchnorm/ReadVariableOp_22j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2b
/batch_normalization_24/batchnorm/ReadVariableOp/batch_normalization_24/batchnorm/ReadVariableOp2f
1batch_normalization_24/batchnorm/ReadVariableOp_11batch_normalization_24/batchnorm/ReadVariableOp_12f
1batch_normalization_24/batchnorm/ReadVariableOp_21batch_normalization_24/batchnorm/ReadVariableOp_22j
3batch_normalization_24/batchnorm/mul/ReadVariableOp3batch_normalization_24/batchnorm/mul/ReadVariableOp2b
/batch_normalization_25/batchnorm/ReadVariableOp/batch_normalization_25/batchnorm/ReadVariableOp2f
1batch_normalization_25/batchnorm/ReadVariableOp_11batch_normalization_25/batchnorm/ReadVariableOp_12f
1batch_normalization_25/batchnorm/ReadVariableOp_21batch_normalization_25/batchnorm/ReadVariableOp_22j
3batch_normalization_25/batchnorm/mul/ReadVariableOp3batch_normalization_25/batchnorm/mul/ReadVariableOp2b
/batch_normalization_26/batchnorm/ReadVariableOp/batch_normalization_26/batchnorm/ReadVariableOp2f
1batch_normalization_26/batchnorm/ReadVariableOp_11batch_normalization_26/batchnorm/ReadVariableOp_12f
1batch_normalization_26/batchnorm/ReadVariableOp_21batch_normalization_26/batchnorm/ReadVariableOp_22j
3batch_normalization_26/batchnorm/mul/ReadVariableOp3batch_normalization_26/batchnorm/mul/ReadVariableOp2b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2f
1batch_normalization_27/batchnorm/ReadVariableOp_11batch_normalization_27/batchnorm/ReadVariableOp_12f
1batch_normalization_27/batchnorm/ReadVariableOp_21batch_normalization_27/batchnorm/ReadVariableOp_22j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2f
1batch_normalization_28/batchnorm/ReadVariableOp_11batch_normalization_28/batchnorm/ReadVariableOp_12f
1batch_normalization_28/batchnorm/ReadVariableOp_21batch_normalization_28/batchnorm/ReadVariableOp_22j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:S O
,
_output_shapes
:??????????

_user_specified_nameinput
?	
?
F__inference_dense_35_layer_call_and_return_conditional_losses_47130198

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?	
?
F__inference_dense_37_layer_call_and_return_conditional_losses_47130497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
f
H__inference_dropout_14_layer_call_and_return_conditional_losses_47130469

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????
2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????
2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_47130264

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:22
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:22
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:22
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:22
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????2::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_28_layer_call_fn_47130452

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:
2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:
2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:
2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
c
G__inference_flatten_8_layer_call_and_return_conditional_losses_47129620

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_47130171

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????F2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????F2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????F:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
??
?
E__inference_model_8_layer_call_and_return_conditional_losses_47129122	
input+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource<
8batch_normalization_23_batchnorm_readvariableop_resource@
<batch_normalization_23_batchnorm_mul_readvariableop_resource>
:batch_normalization_23_batchnorm_readvariableop_1_resource>
:batch_normalization_23_batchnorm_readvariableop_2_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource<
8batch_normalization_24_batchnorm_readvariableop_resource@
<batch_normalization_24_batchnorm_mul_readvariableop_resource>
:batch_normalization_24_batchnorm_readvariableop_1_resource>
:batch_normalization_24_batchnorm_readvariableop_2_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource<
8batch_normalization_25_batchnorm_readvariableop_resource@
<batch_normalization_25_batchnorm_mul_readvariableop_resource>
:batch_normalization_25_batchnorm_readvariableop_1_resource>
:batch_normalization_25_batchnorm_readvariableop_2_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource<
8batch_normalization_26_batchnorm_readvariableop_resource@
<batch_normalization_26_batchnorm_mul_readvariableop_resource>
:batch_normalization_26_batchnorm_readvariableop_1_resource>
:batch_normalization_26_batchnorm_readvariableop_2_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource<
8batch_normalization_27_batchnorm_readvariableop_resource@
<batch_normalization_27_batchnorm_mul_readvariableop_resource>
:batch_normalization_27_batchnorm_readvariableop_1_resource>
:batch_normalization_27_batchnorm_readvariableop_2_resource+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource<
8batch_normalization_28_batchnorm_readvariableop_resource@
<batch_normalization_28_batchnorm_mul_readvariableop_resource>
:batch_normalization_28_batchnorm_readvariableop_1_resource>
:batch_normalization_28_batchnorm_readvariableop_2_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??/batch_normalization_23/batchnorm/ReadVariableOp?1batch_normalization_23/batchnorm/ReadVariableOp_1?1batch_normalization_23/batchnorm/ReadVariableOp_2?3batch_normalization_23/batchnorm/mul/ReadVariableOp?/batch_normalization_24/batchnorm/ReadVariableOp?1batch_normalization_24/batchnorm/ReadVariableOp_1?1batch_normalization_24/batchnorm/ReadVariableOp_2?3batch_normalization_24/batchnorm/mul/ReadVariableOp?/batch_normalization_25/batchnorm/ReadVariableOp?1batch_normalization_25/batchnorm/ReadVariableOp_1?1batch_normalization_25/batchnorm/ReadVariableOp_2?3batch_normalization_25/batchnorm/mul/ReadVariableOp?/batch_normalization_26/batchnorm/ReadVariableOp?1batch_normalization_26/batchnorm/ReadVariableOp_1?1batch_normalization_26/batchnorm/ReadVariableOp_2?3batch_normalization_26/batchnorm/mul/ReadVariableOp?/batch_normalization_27/batchnorm/ReadVariableOp?1batch_normalization_27/batchnorm/ReadVariableOp_1?1batch_normalization_27/batchnorm/ReadVariableOp_2?3batch_normalization_27/batchnorm/mul/ReadVariableOp?/batch_normalization_28/batchnorm/ReadVariableOp?1batch_normalization_28/batchnorm/ReadVariableOp_1?1batch_normalization_28/batchnorm/ReadVariableOp_2?3batch_normalization_28/batchnorm/mul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?dense_36/BiasAdd/ReadVariableOp?dense_36/MatMul/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?dense_37/MatMul/ReadVariableOps
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_8/Const?
flatten_8/ReshapeReshapeinputflatten_8/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_8/Reshape?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMulflatten_8/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd?
/batch_normalization_23/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_23_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_23/batchnorm/ReadVariableOp?
&batch_normalization_23/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_23/batchnorm/add/y?
$batch_normalization_23/batchnorm/addAddV27batch_normalization_23/batchnorm/ReadVariableOp:value:0/batch_normalization_23/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/add?
&batch_normalization_23/batchnorm/RsqrtRsqrt(batch_normalization_23/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/Rsqrt?
3batch_normalization_23/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_23_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_23/batchnorm/mul/ReadVariableOp?
$batch_normalization_23/batchnorm/mulMul*batch_normalization_23/batchnorm/Rsqrt:y:0;batch_normalization_23/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/mul?
&batch_normalization_23/batchnorm/mul_1Muldense_31/BiasAdd:output:0(batch_normalization_23/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/mul_1?
1batch_normalization_23/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_1?
&batch_normalization_23/batchnorm/mul_2Mul9batch_normalization_23/batchnorm/ReadVariableOp_1:value:0(batch_normalization_23/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_23/batchnorm/mul_2?
1batch_normalization_23/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_23_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_23/batchnorm/ReadVariableOp_2?
$batch_normalization_23/batchnorm/subSub9batch_normalization_23/batchnorm/ReadVariableOp_2:value:0*batch_normalization_23/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_23/batchnorm/sub?
&batch_normalization_23/batchnorm/add_1AddV2*batch_normalization_23/batchnorm/mul_1:z:0(batch_normalization_23/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_23/batchnorm/add_1?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMul*batch_normalization_23/batchnorm/add_1:z:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd?
/batch_normalization_24/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_24_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype021
/batch_normalization_24/batchnorm/ReadVariableOp?
&batch_normalization_24/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_24/batchnorm/add/y?
$batch_normalization_24/batchnorm/addAddV27batch_normalization_24/batchnorm/ReadVariableOp:value:0/batch_normalization_24/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/add?
&batch_normalization_24/batchnorm/RsqrtRsqrt(batch_normalization_24/batchnorm/add:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/Rsqrt?
3batch_normalization_24/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_24_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype025
3batch_normalization_24/batchnorm/mul/ReadVariableOp?
$batch_normalization_24/batchnorm/mulMul*batch_normalization_24/batchnorm/Rsqrt:y:0;batch_normalization_24/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/mul?
&batch_normalization_24/batchnorm/mul_1Muldense_32/BiasAdd:output:0(batch_normalization_24/batchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/mul_1?
1batch_normalization_24/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_1?
&batch_normalization_24/batchnorm/mul_2Mul9batch_normalization_24/batchnorm/ReadVariableOp_1:value:0(batch_normalization_24/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2(
&batch_normalization_24/batchnorm/mul_2?
1batch_normalization_24/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_24_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype023
1batch_normalization_24/batchnorm/ReadVariableOp_2?
$batch_normalization_24/batchnorm/subSub9batch_normalization_24/batchnorm/ReadVariableOp_2:value:0*batch_normalization_24/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2&
$batch_normalization_24/batchnorm/sub?
&batch_normalization_24/batchnorm/add_1AddV2*batch_normalization_24/batchnorm/mul_1:z:0(batch_normalization_24/batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2(
&batch_normalization_24/batchnorm/add_1?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMul*batch_normalization_24/batchnorm/add_1:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_33/BiasAdd?
/batch_normalization_25/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_25_batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype021
/batch_normalization_25/batchnorm/ReadVariableOp?
&batch_normalization_25/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_25/batchnorm/add/y?
$batch_normalization_25/batchnorm/addAddV27batch_normalization_25/batchnorm/ReadVariableOp:value:0/batch_normalization_25/batchnorm/add/y:output:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/add?
&batch_normalization_25/batchnorm/RsqrtRsqrt(batch_normalization_25/batchnorm/add:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/Rsqrt?
3batch_normalization_25/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_25_batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype025
3batch_normalization_25/batchnorm/mul/ReadVariableOp?
$batch_normalization_25/batchnorm/mulMul*batch_normalization_25/batchnorm/Rsqrt:y:0;batch_normalization_25/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/mul?
&batch_normalization_25/batchnorm/mul_1Muldense_33/BiasAdd:output:0(batch_normalization_25/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/mul_1?
1batch_normalization_25/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_1_resource*
_output_shapes
:d*
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_1?
&batch_normalization_25/batchnorm/mul_2Mul9batch_normalization_25/batchnorm/ReadVariableOp_1:value:0(batch_normalization_25/batchnorm/mul:z:0*
T0*
_output_shapes
:d2(
&batch_normalization_25/batchnorm/mul_2?
1batch_normalization_25/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_25_batchnorm_readvariableop_2_resource*
_output_shapes
:d*
dtype023
1batch_normalization_25/batchnorm/ReadVariableOp_2?
$batch_normalization_25/batchnorm/subSub9batch_normalization_25/batchnorm/ReadVariableOp_2:value:0*batch_normalization_25/batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2&
$batch_normalization_25/batchnorm/sub?
&batch_normalization_25/batchnorm/add_1AddV2*batch_normalization_25/batchnorm/mul_1:z:0(batch_normalization_25/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2(
&batch_normalization_25/batchnorm/add_1?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:dF*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul*batch_normalization_25/batchnorm/add_1:z:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:F*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
dense_34/BiasAdd?
/batch_normalization_26/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype021
/batch_normalization_26/batchnorm/ReadVariableOp?
&batch_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_26/batchnorm/add/y?
$batch_normalization_26/batchnorm/addAddV27batch_normalization_26/batchnorm/ReadVariableOp:value:0/batch_normalization_26/batchnorm/add/y:output:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/add?
&batch_normalization_26/batchnorm/RsqrtRsqrt(batch_normalization_26/batchnorm/add:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/Rsqrt?
3batch_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype025
3batch_normalization_26/batchnorm/mul/ReadVariableOp?
$batch_normalization_26/batchnorm/mulMul*batch_normalization_26/batchnorm/Rsqrt:y:0;batch_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/mul?
&batch_normalization_26/batchnorm/mul_1Muldense_34/BiasAdd:output:0(batch_normalization_26/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/mul_1?
1batch_normalization_26/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_1?
&batch_normalization_26/batchnorm/mul_2Mul9batch_normalization_26/batchnorm/ReadVariableOp_1:value:0(batch_normalization_26/batchnorm/mul:z:0*
T0*
_output_shapes
:F2(
&batch_normalization_26/batchnorm/mul_2?
1batch_normalization_26/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_26_batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype023
1batch_normalization_26/batchnorm/ReadVariableOp_2?
$batch_normalization_26/batchnorm/subSub9batch_normalization_26/batchnorm/ReadVariableOp_2:value:0*batch_normalization_26/batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2&
$batch_normalization_26/batchnorm/sub?
&batch_normalization_26/batchnorm/add_1AddV2*batch_normalization_26/batchnorm/mul_1:z:0(batch_normalization_26/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2(
&batch_normalization_26/batchnorm/add_1?
dropout_13/IdentityIdentity*batch_normalization_26/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????F2
dropout_13/Identity?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:F2*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldropout_13/Identity:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense_35/BiasAdd?
/batch_normalization_27/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:2*
dtype021
/batch_normalization_27/batchnorm/ReadVariableOp?
&batch_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_27/batchnorm/add/y?
$batch_normalization_27/batchnorm/addAddV27batch_normalization_27/batchnorm/ReadVariableOp:value:0/batch_normalization_27/batchnorm/add/y:output:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/add?
&batch_normalization_27/batchnorm/RsqrtRsqrt(batch_normalization_27/batchnorm/add:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/Rsqrt?
3batch_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:2*
dtype025
3batch_normalization_27/batchnorm/mul/ReadVariableOp?
$batch_normalization_27/batchnorm/mulMul*batch_normalization_27/batchnorm/Rsqrt:y:0;batch_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/mul?
&batch_normalization_27/batchnorm/mul_1Muldense_35/BiasAdd:output:0(batch_normalization_27/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/mul_1?
1batch_normalization_27/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_1_resource*
_output_shapes
:2*
dtype023
1batch_normalization_27/batchnorm/ReadVariableOp_1?
&batch_normalization_27/batchnorm/mul_2Mul9batch_normalization_27/batchnorm/ReadVariableOp_1:value:0(batch_normalization_27/batchnorm/mul:z:0*
T0*
_output_shapes
:22(
&batch_normalization_27/batchnorm/mul_2?
1batch_normalization_27/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_27_batchnorm_readvariableop_2_resource*
_output_shapes
:2*
dtype023
1batch_normalization_27/batchnorm/ReadVariableOp_2?
$batch_normalization_27/batchnorm/subSub9batch_normalization_27/batchnorm/ReadVariableOp_2:value:0*batch_normalization_27/batchnorm/mul_2:z:0*
T0*
_output_shapes
:22&
$batch_normalization_27/batchnorm/sub?
&batch_normalization_27/batchnorm/add_1AddV2*batch_normalization_27/batchnorm/mul_1:z:0(batch_normalization_27/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????22(
&batch_normalization_27/batchnorm/add_1?
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMul*batch_normalization_27/batchnorm/add_1:z:0&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_36/BiasAdd?
/batch_normalization_28/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_28_batchnorm_readvariableop_resource*
_output_shapes
:
*
dtype021
/batch_normalization_28/batchnorm/ReadVariableOp?
&batch_normalization_28/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2(
&batch_normalization_28/batchnorm/add/y?
$batch_normalization_28/batchnorm/addAddV27batch_normalization_28/batchnorm/ReadVariableOp:value:0/batch_normalization_28/batchnorm/add/y:output:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/add?
&batch_normalization_28/batchnorm/RsqrtRsqrt(batch_normalization_28/batchnorm/add:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/Rsqrt?
3batch_normalization_28/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_28_batchnorm_mul_readvariableop_resource*
_output_shapes
:
*
dtype025
3batch_normalization_28/batchnorm/mul/ReadVariableOp?
$batch_normalization_28/batchnorm/mulMul*batch_normalization_28/batchnorm/Rsqrt:y:0;batch_normalization_28/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/mul?
&batch_normalization_28/batchnorm/mul_1Muldense_36/BiasAdd:output:0(batch_normalization_28/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/mul_1?
1batch_normalization_28/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_1_resource*
_output_shapes
:
*
dtype023
1batch_normalization_28/batchnorm/ReadVariableOp_1?
&batch_normalization_28/batchnorm/mul_2Mul9batch_normalization_28/batchnorm/ReadVariableOp_1:value:0(batch_normalization_28/batchnorm/mul:z:0*
T0*
_output_shapes
:
2(
&batch_normalization_28/batchnorm/mul_2?
1batch_normalization_28/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_28_batchnorm_readvariableop_2_resource*
_output_shapes
:
*
dtype023
1batch_normalization_28/batchnorm/ReadVariableOp_2?
$batch_normalization_28/batchnorm/subSub9batch_normalization_28/batchnorm/ReadVariableOp_2:value:0*batch_normalization_28/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
2&
$batch_normalization_28/batchnorm/sub?
&batch_normalization_28/batchnorm/add_1AddV2*batch_normalization_28/batchnorm/mul_1:z:0(batch_normalization_28/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????
2(
&batch_normalization_28/batchnorm/add_1?
dropout_14/IdentityIdentity*batch_normalization_28/batchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????
2
dropout_14/Identity?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldropout_14/Identity:output:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdd|
dense_37/SoftmaxSoftmaxdense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Softmax?
IdentityIdentitydense_37/Softmax:softmax:00^batch_normalization_23/batchnorm/ReadVariableOp2^batch_normalization_23/batchnorm/ReadVariableOp_12^batch_normalization_23/batchnorm/ReadVariableOp_24^batch_normalization_23/batchnorm/mul/ReadVariableOp0^batch_normalization_24/batchnorm/ReadVariableOp2^batch_normalization_24/batchnorm/ReadVariableOp_12^batch_normalization_24/batchnorm/ReadVariableOp_24^batch_normalization_24/batchnorm/mul/ReadVariableOp0^batch_normalization_25/batchnorm/ReadVariableOp2^batch_normalization_25/batchnorm/ReadVariableOp_12^batch_normalization_25/batchnorm/ReadVariableOp_24^batch_normalization_25/batchnorm/mul/ReadVariableOp0^batch_normalization_26/batchnorm/ReadVariableOp2^batch_normalization_26/batchnorm/ReadVariableOp_12^batch_normalization_26/batchnorm/ReadVariableOp_24^batch_normalization_26/batchnorm/mul/ReadVariableOp0^batch_normalization_27/batchnorm/ReadVariableOp2^batch_normalization_27/batchnorm/ReadVariableOp_12^batch_normalization_27/batchnorm/ReadVariableOp_24^batch_normalization_27/batchnorm/mul/ReadVariableOp0^batch_normalization_28/batchnorm/ReadVariableOp2^batch_normalization_28/batchnorm/ReadVariableOp_12^batch_normalization_28/batchnorm/ReadVariableOp_24^batch_normalization_28/batchnorm/mul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp ^dense_36/BiasAdd/ReadVariableOp^dense_36/MatMul/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp^dense_37/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????::::::::::::::::::::::::::::::::::::::2b
/batch_normalization_23/batchnorm/ReadVariableOp/batch_normalization_23/batchnorm/ReadVariableOp2f
1batch_normalization_23/batchnorm/ReadVariableOp_11batch_normalization_23/batchnorm/ReadVariableOp_12f
1batch_normalization_23/batchnorm/ReadVariableOp_21batch_normalization_23/batchnorm/ReadVariableOp_22j
3batch_normalization_23/batchnorm/mul/ReadVariableOp3batch_normalization_23/batchnorm/mul/ReadVariableOp2b
/batch_normalization_24/batchnorm/ReadVariableOp/batch_normalization_24/batchnorm/ReadVariableOp2f
1batch_normalization_24/batchnorm/ReadVariableOp_11batch_normalization_24/batchnorm/ReadVariableOp_12f
1batch_normalization_24/batchnorm/ReadVariableOp_21batch_normalization_24/batchnorm/ReadVariableOp_22j
3batch_normalization_24/batchnorm/mul/ReadVariableOp3batch_normalization_24/batchnorm/mul/ReadVariableOp2b
/batch_normalization_25/batchnorm/ReadVariableOp/batch_normalization_25/batchnorm/ReadVariableOp2f
1batch_normalization_25/batchnorm/ReadVariableOp_11batch_normalization_25/batchnorm/ReadVariableOp_12f
1batch_normalization_25/batchnorm/ReadVariableOp_21batch_normalization_25/batchnorm/ReadVariableOp_22j
3batch_normalization_25/batchnorm/mul/ReadVariableOp3batch_normalization_25/batchnorm/mul/ReadVariableOp2b
/batch_normalization_26/batchnorm/ReadVariableOp/batch_normalization_26/batchnorm/ReadVariableOp2f
1batch_normalization_26/batchnorm/ReadVariableOp_11batch_normalization_26/batchnorm/ReadVariableOp_12f
1batch_normalization_26/batchnorm/ReadVariableOp_21batch_normalization_26/batchnorm/ReadVariableOp_22j
3batch_normalization_26/batchnorm/mul/ReadVariableOp3batch_normalization_26/batchnorm/mul/ReadVariableOp2b
/batch_normalization_27/batchnorm/ReadVariableOp/batch_normalization_27/batchnorm/ReadVariableOp2f
1batch_normalization_27/batchnorm/ReadVariableOp_11batch_normalization_27/batchnorm/ReadVariableOp_12f
1batch_normalization_27/batchnorm/ReadVariableOp_21batch_normalization_27/batchnorm/ReadVariableOp_22j
3batch_normalization_27/batchnorm/mul/ReadVariableOp3batch_normalization_27/batchnorm/mul/ReadVariableOp2b
/batch_normalization_28/batchnorm/ReadVariableOp/batch_normalization_28/batchnorm/ReadVariableOp2f
1batch_normalization_28/batchnorm/ReadVariableOp_11batch_normalization_28/batchnorm/ReadVariableOp_12f
1batch_normalization_28/batchnorm/ReadVariableOp_21batch_normalization_28/batchnorm/ReadVariableOp_22j
3batch_normalization_28/batchnorm/mul/ReadVariableOp3batch_normalization_28/batchnorm/mul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2@
dense_36/MatMul/ReadVariableOpdense_36/MatMul/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2@
dense_37/MatMul/ReadVariableOpdense_37/MatMul/ReadVariableOp:S O
,
_output_shapes
:??????????

_user_specified_nameinput
?
?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_47130098

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:F2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:F2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:F*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:F2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:F2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:F*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:F2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????F2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?0
?
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_47129814

inputs
assignmovingavg_47129789
assignmovingavg_1_47129795)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:??????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129789*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47129789*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129789*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129789*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47129789AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129789*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129795*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47129795*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129795*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129795*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47129795AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129795*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:??????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dense_34_layer_call_fn_47130042

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dF*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:F*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????F2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????F2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
F__inference_dense_32_layer_call_and_return_conditional_losses_47129768

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?0
?
9__inference_batch_normalization_25_layer_call_fn_47130002

inputs
assignmovingavg_47129977
assignmovingavg_1_47129983)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:d2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????d2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:d*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129977*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_47129977*
_output_shapes
:d*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129977*
_output_shapes
:d2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@AssignMovingAvg/47129977*
_output_shapes
:d2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_47129977AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*+
_class!
loc:@AssignMovingAvg/47129977*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129983*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_47129983*
_output_shapes
:d*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129983*
_output_shapes
:d2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/47129983*
_output_shapes
:d2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_47129983AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*-
_class#
!loc:@AssignMovingAvg_1/47129983*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:d2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:d2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:d2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:d*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:d2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????d2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????d::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs"?-
saver_filename:0
Identity:0Identity_468"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input3
serving_default_input:0??????????<
dense_370
StatefulPartitionedCall:0?????????tensorflow/serving/predict:͑
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer-15
layer_with_weights-12
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "model_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["flatten_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["dense_31", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_24", "inbound_nodes": [[["dense_32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["batch_normalization_24", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_25", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 70, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["batch_normalization_25", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_26", "inbound_nodes": [[["dense_34", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["batch_normalization_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_27", "inbound_nodes": [[["dense_35", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_36", "inbound_nodes": [[["batch_normalization_27", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_28", "inbound_nodes": [[["dense_36", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_14", "inbound_nodes": [[["batch_normalization_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_37", "inbound_nodes": [[["dropout_14", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense_37", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 6, 200]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 200]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_8", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["flatten_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["dense_31", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_24", "inbound_nodes": [[["dense_32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["batch_normalization_24", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_25", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 70, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["batch_normalization_25", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_26", "inbound_nodes": [[["dense_34", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_13", "inbound_nodes": [[["batch_normalization_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dropout_13", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_27", "inbound_nodes": [[["dense_35", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_36", "inbound_nodes": [[["batch_normalization_27", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_28", "inbound_nodes": [[["dense_36", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_14", "inbound_nodes": [[["batch_normalization_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_37", "inbound_nodes": [[["dropout_14", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["dense_37", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": {"class_name": "ExponentialDecay", "config": {"initial_learning_rate": 0.001, "decay_steps": 300, "decay_rate": 0.1, "staircase": false, "name": null}}, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 200]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
trainable_variables
	variables
 regularization_losses
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1200]}}
?	
"axis
	#gamma
$beta
%moving_mean
&moving_variance
'trainable_variables
(	variables
)regularization_losses
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?

+kernel
,bias
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 300, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
?	
1axis
	2gamma
3beta
4moving_mean
5moving_variance
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_24", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
?	
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_25", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?

Ikernel
Jbias
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 70, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?	
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_26", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 70}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}}
?
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

\kernel
]bias
^trainable_variables
_	variables
`regularization_losses
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}}
?	
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_27", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_27", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?

kkernel
lbias
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?	
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_28", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

~kernel
bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
9
	?iter

?decay
?momentum"
	optimizer
?
0
1
#2
$3
+4
,5
26
37
:8
;9
A10
B11
I12
J13
P14
Q15
\16
]17
c18
d19
k20
l21
r22
s23
~24
25"
trackable_list_wrapper
?
0
1
#2
$3
%4
&5
+6
,7
28
39
410
511
:12
;13
A14
B15
C16
D17
I18
J19
P20
Q21
R22
S23
\24
]25
c26
d27
e28
f29
k30
l31
r32
s33
t34
u35
~36
37"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?non_trainable_variables
	variables
?layer_metrics
?layers
regularization_losses
 ?layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?non_trainable_variables
	variables
?layer_metrics
?layers
regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
?	?2dense_31/kernel
:?2dense_31/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?metrics
?non_trainable_variables
	variables
?layer_metrics
?layers
 regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_23/gamma
*:(?2batch_normalization_23/beta
3:1? (2"batch_normalization_23/moving_mean
7:5? (2&batch_normalization_23/moving_variance
.
#0
$1"
trackable_list_wrapper
<
#0
$1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
'trainable_variables
?metrics
?non_trainable_variables
(	variables
?layer_metrics
?layers
)regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_32/kernel
:?2dense_32/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-trainable_variables
?metrics
?non_trainable_variables
.	variables
?layer_metrics
?layers
/regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_24/gamma
*:(?2batch_normalization_24/beta
3:1? (2"batch_normalization_24/moving_mean
7:5? (2&batch_normalization_24/moving_variance
.
20
31"
trackable_list_wrapper
<
20
31
42
53"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6trainable_variables
?metrics
?non_trainable_variables
7	variables
?layer_metrics
?layers
8regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?d2dense_33/kernel
:d2dense_33/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<trainable_variables
?metrics
?non_trainable_variables
=	variables
?layer_metrics
?layers
>regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(d2batch_normalization_25/gamma
):'d2batch_normalization_25/beta
2:0d (2"batch_normalization_25/moving_mean
6:4d (2&batch_normalization_25/moving_variance
.
A0
B1"
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Etrainable_variables
?metrics
?non_trainable_variables
F	variables
?layer_metrics
?layers
Gregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:dF2dense_34/kernel
:F2dense_34/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ktrainable_variables
?metrics
?non_trainable_variables
L	variables
?layer_metrics
?layers
Mregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(F2batch_normalization_26/gamma
):'F2batch_normalization_26/beta
2:0F (2"batch_normalization_26/moving_mean
6:4F (2&batch_normalization_26/moving_variance
.
P0
Q1"
trackable_list_wrapper
<
P0
Q1
R2
S3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ttrainable_variables
?metrics
?non_trainable_variables
U	variables
?layer_metrics
?layers
Vregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xtrainable_variables
?metrics
?non_trainable_variables
Y	variables
?layer_metrics
?layers
Zregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:F22dense_35/kernel
:22dense_35/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^trainable_variables
?metrics
?non_trainable_variables
_	variables
?layer_metrics
?layers
`regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(22batch_normalization_27/gamma
):'22batch_normalization_27/beta
2:02 (2"batch_normalization_27/moving_mean
6:42 (2&batch_normalization_27/moving_variance
.
c0
d1"
trackable_list_wrapper
<
c0
d1
e2
f3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gtrainable_variables
?metrics
?non_trainable_variables
h	variables
?layer_metrics
?layers
iregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2
2dense_36/kernel
:
2dense_36/bias
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mtrainable_variables
?metrics
?non_trainable_variables
n	variables
?layer_metrics
?layers
oregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(
2batch_normalization_28/gamma
):'
2batch_normalization_28/beta
2:0
 (2"batch_normalization_28/moving_mean
6:4
 (2&batch_normalization_28/moving_variance
.
r0
s1"
trackable_list_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vtrainable_variables
?metrics
?non_trainable_variables
w	variables
?layer_metrics
?layers
xregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ztrainable_variables
?metrics
?non_trainable_variables
{	variables
?layer_metrics
?layers
|regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:
2dense_37/kernel
:2dense_37/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?metrics
?non_trainable_variables
?	variables
?layer_metrics
?layers
?regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/momentum
0
?0
?1"
trackable_list_wrapper
v
%0
&1
42
53
C4
D5
R6
S7
e8
f9
t10
u11"
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
E__inference_model_8_layer_call_and_return_conditional_losses_47129122
E__inference_model_8_layer_call_and_return_conditional_losses_47128975?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_47127119?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *)?&
$?!
input??????????
?2?
*__inference_model_8_layer_call_fn_47129527
*__inference_model_8_layer_call_fn_47129380?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_flatten_8_layer_call_and_return_conditional_losses_47129620?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_8_layer_call_fn_47129626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_31_layer_call_and_return_conditional_losses_47129636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_31_layer_call_fn_47129646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47129702
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47129682?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_23_layer_call_fn_47129738
9__inference_batch_normalization_23_layer_call_fn_47129758?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_32_layer_call_and_return_conditional_losses_47129768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_32_layer_call_fn_47129778?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_47129814
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_47129834?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_24_layer_call_fn_47129890
9__inference_batch_normalization_24_layer_call_fn_47129870?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_33_layer_call_and_return_conditional_losses_47129900?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_33_layer_call_fn_47129910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_47129946
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_47129966?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_25_layer_call_fn_47130002
9__inference_batch_normalization_25_layer_call_fn_47130022?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_34_layer_call_and_return_conditional_losses_47130032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_34_layer_call_fn_47130042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_47130078
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_47130098?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_26_layer_call_fn_47130154
9__inference_batch_normalization_26_layer_call_fn_47130134?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_13_layer_call_and_return_conditional_losses_47130171
H__inference_dropout_13_layer_call_and_return_conditional_losses_47130166?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_13_layer_call_fn_47130183
-__inference_dropout_13_layer_call_fn_47130188?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_35_layer_call_and_return_conditional_losses_47130198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_35_layer_call_fn_47130208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_47130264
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_47130244?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_27_layer_call_fn_47130320
9__inference_batch_normalization_27_layer_call_fn_47130300?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_36_layer_call_and_return_conditional_losses_47130330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_36_layer_call_fn_47130340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_47130396
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_47130376?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
9__inference_batch_normalization_28_layer_call_fn_47130432
9__inference_batch_normalization_28_layer_call_fn_47130452?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_14_layer_call_and_return_conditional_losses_47130464
H__inference_dropout_14_layer_call_and_return_conditional_losses_47130469?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_14_layer_call_fn_47130481
-__inference_dropout_14_layer_call_fn_47130486?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dense_37_layer_call_and_return_conditional_losses_47130497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_37_layer_call_fn_47130508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_47129614input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_47127119?&&#%$+,5243:;DACBIJSPRQ\]fcedklurts~3?0
)?&
$?!
input??????????
? "3?0
.
dense_37"?
dense_37??????????
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47129682d%&#$4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_47129702d&#%$4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
9__inference_batch_normalization_23_layer_call_fn_47129738W%&#$4?1
*?'
!?
inputs??????????
p
? "????????????
9__inference_batch_normalization_23_layer_call_fn_47129758W&#%$4?1
*?'
!?
inputs??????????
p 
? "????????????
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_47129814d45234?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
T__inference_batch_normalization_24_layer_call_and_return_conditional_losses_47129834d52434?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
9__inference_batch_normalization_24_layer_call_fn_47129870W45234?1
*?'
!?
inputs??????????
p
? "????????????
9__inference_batch_normalization_24_layer_call_fn_47129890W52434?1
*?'
!?
inputs??????????
p 
? "????????????
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_47129946bCDAB3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
T__inference_batch_normalization_25_layer_call_and_return_conditional_losses_47129966bDACB3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
9__inference_batch_normalization_25_layer_call_fn_47130002UCDAB3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
9__inference_batch_normalization_25_layer_call_fn_47130022UDACB3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_47130078bRSPQ3?0
)?&
 ?
inputs?????????F
p
? "%?"
?
0?????????F
? ?
T__inference_batch_normalization_26_layer_call_and_return_conditional_losses_47130098bSPRQ3?0
)?&
 ?
inputs?????????F
p 
? "%?"
?
0?????????F
? ?
9__inference_batch_normalization_26_layer_call_fn_47130134URSPQ3?0
)?&
 ?
inputs?????????F
p
? "??????????F?
9__inference_batch_normalization_26_layer_call_fn_47130154USPRQ3?0
)?&
 ?
inputs?????????F
p 
? "??????????F?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_47130244befcd3?0
)?&
 ?
inputs?????????2
p
? "%?"
?
0?????????2
? ?
T__inference_batch_normalization_27_layer_call_and_return_conditional_losses_47130264bfced3?0
)?&
 ?
inputs?????????2
p 
? "%?"
?
0?????????2
? ?
9__inference_batch_normalization_27_layer_call_fn_47130300Uefcd3?0
)?&
 ?
inputs?????????2
p
? "??????????2?
9__inference_batch_normalization_27_layer_call_fn_47130320Ufced3?0
)?&
 ?
inputs?????????2
p 
? "??????????2?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_47130376bturs3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
T__inference_batch_normalization_28_layer_call_and_return_conditional_losses_47130396burts3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
9__inference_batch_normalization_28_layer_call_fn_47130432Uturs3?0
)?&
 ?
inputs?????????

p
? "??????????
?
9__inference_batch_normalization_28_layer_call_fn_47130452Uurts3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
F__inference_dense_31_layer_call_and_return_conditional_losses_47129636^0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? ?
+__inference_dense_31_layer_call_fn_47129646Q0?-
&?#
!?
inputs??????????	
? "????????????
F__inference_dense_32_layer_call_and_return_conditional_losses_47129768^+,0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_32_layer_call_fn_47129778Q+,0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_33_layer_call_and_return_conditional_losses_47129900]:;0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? 
+__inference_dense_33_layer_call_fn_47129910P:;0?-
&?#
!?
inputs??????????
? "??????????d?
F__inference_dense_34_layer_call_and_return_conditional_losses_47130032\IJ/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????F
? ~
+__inference_dense_34_layer_call_fn_47130042OIJ/?,
%?"
 ?
inputs?????????d
? "??????????F?
F__inference_dense_35_layer_call_and_return_conditional_losses_47130198\\]/?,
%?"
 ?
inputs?????????F
? "%?"
?
0?????????2
? ~
+__inference_dense_35_layer_call_fn_47130208O\]/?,
%?"
 ?
inputs?????????F
? "??????????2?
F__inference_dense_36_layer_call_and_return_conditional_losses_47130330\kl/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????

? ~
+__inference_dense_36_layer_call_fn_47130340Okl/?,
%?"
 ?
inputs?????????2
? "??????????
?
F__inference_dense_37_layer_call_and_return_conditional_losses_47130497\~/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ~
+__inference_dense_37_layer_call_fn_47130508O~/?,
%?"
 ?
inputs?????????

? "???????????
H__inference_dropout_13_layer_call_and_return_conditional_losses_47130166\3?0
)?&
 ?
inputs?????????F
p
? "%?"
?
0?????????F
? ?
H__inference_dropout_13_layer_call_and_return_conditional_losses_47130171\3?0
)?&
 ?
inputs?????????F
p 
? "%?"
?
0?????????F
? ?
-__inference_dropout_13_layer_call_fn_47130183O3?0
)?&
 ?
inputs?????????F
p
? "??????????F?
-__inference_dropout_13_layer_call_fn_47130188O3?0
)?&
 ?
inputs?????????F
p 
? "??????????F?
H__inference_dropout_14_layer_call_and_return_conditional_losses_47130464\3?0
)?&
 ?
inputs?????????

p
? "%?"
?
0?????????

? ?
H__inference_dropout_14_layer_call_and_return_conditional_losses_47130469\3?0
)?&
 ?
inputs?????????

p 
? "%?"
?
0?????????

? ?
-__inference_dropout_14_layer_call_fn_47130481O3?0
)?&
 ?
inputs?????????

p
? "??????????
?
-__inference_dropout_14_layer_call_fn_47130486O3?0
)?&
 ?
inputs?????????

p 
? "??????????
?
G__inference_flatten_8_layer_call_and_return_conditional_losses_47129620^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????	
? ?
,__inference_flatten_8_layer_call_fn_47129626Q4?1
*?'
%?"
inputs??????????
? "???????????	?
E__inference_model_8_layer_call_and_return_conditional_losses_47128975?&%&#$+,4523:;CDABIJRSPQ\]efcdklturs~;?8
1?.
$?!
input??????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_8_layer_call_and_return_conditional_losses_47129122?&&#%$+,5243:;DACBIJSPRQ\]fcedklurts~;?8
1?.
$?!
input??????????
p 

 
? "%?"
?
0?????????
? ?
*__inference_model_8_layer_call_fn_47129380&%&#$+,4523:;CDABIJRSPQ\]efcdklturs~;?8
1?.
$?!
input??????????
p

 
? "???????????
*__inference_model_8_layer_call_fn_47129527&&#%$+,5243:;DACBIJSPRQ\]fcedklurts~;?8
1?.
$?!
input??????????
p 

 
? "???????????
&__inference_signature_wrapper_47129614?&&#%$+,5243:;DACBIJSPRQ\]fcedklurts~<?9
? 
2?/
-
input$?!
input??????????"3?0
.
dense_37"?
dense_37?????????