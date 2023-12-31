�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
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
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0�
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
list(type)(0�
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
executor_typestring �
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.22v2.6.1-9-gc2363d6d0258��

�
layer_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namelayer_conv2/kernel
}
&layer_conv2/kernel/Read/ReadVariableOpReadVariableOplayer_conv2/kernel*"
_output_shapes
: *
dtype0
x
layer_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namelayer_conv2/bias
q
$layer_conv2/bias/Read/ReadVariableOpReadVariableOplayer_conv2/bias*
_output_shapes
: *
dtype0
�
batch_normalization_61/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_61/gamma
�
0batch_normalization_61/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_61/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_61/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_61/beta
�
/batch_normalization_61/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_61/beta*
_output_shapes
: *
dtype0
�
"batch_normalization_61/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_61/moving_mean
�
6batch_normalization_61/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_61/moving_mean*
_output_shapes
: *
dtype0
�
&batch_normalization_61/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_61/moving_variance
�
:batch_normalization_61/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_61/moving_variance*
_output_shapes
: *
dtype0
q

fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�(@*
shared_name
fc1/kernel
j
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel*
_output_shapes
:	�(@*
dtype0
h
fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
fc1/bias
a
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
_output_shapes
:@*
dtype0
p

fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
fc2/kernel
i
fc2/kernel/Read/ReadVariableOpReadVariableOp
fc2/kernel*
_output_shapes

:@*
dtype0
h
fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fc2/bias
a
fc2/bias/Read/ReadVariableOpReadVariableOpfc2/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
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
�
SGD/layer_conv2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!SGD/layer_conv2/kernel/momentum
�
3SGD/layer_conv2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/layer_conv2/kernel/momentum*"
_output_shapes
: *
dtype0
�
SGD/layer_conv2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameSGD/layer_conv2/bias/momentum
�
1SGD/layer_conv2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/layer_conv2/bias/momentum*
_output_shapes
: *
dtype0
�
)SGD/batch_normalization_61/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)SGD/batch_normalization_61/gamma/momentum
�
=SGD/batch_normalization_61/gamma/momentum/Read/ReadVariableOpReadVariableOp)SGD/batch_normalization_61/gamma/momentum*
_output_shapes
: *
dtype0
�
(SGD/batch_normalization_61/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_61/beta/momentum
�
<SGD/batch_normalization_61/beta/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_61/beta/momentum*
_output_shapes
: *
dtype0
�
SGD/fc1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�(@*(
shared_nameSGD/fc1/kernel/momentum
�
+SGD/fc1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/fc1/kernel/momentum*
_output_shapes
:	�(@*
dtype0
�
SGD/fc1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameSGD/fc1/bias/momentum
{
)SGD/fc1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/fc1/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/fc2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameSGD/fc2/kernel/momentum
�
+SGD/fc2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/fc2/kernel/momentum*
_output_shapes

:@*
dtype0
�
SGD/fc2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameSGD/fc2/bias/momentum
{
)SGD/fc2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/fc2/bias/momentum*
_output_shapes
:*
dtype0

NoOpNoOp
�4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�4
value�4B�4 B�4
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
R
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
�
	@decay
Alearning_rate
Bmomentum
Citermomentum�momentum�momentum�momentum�0momentum�1momentum�:momentum�;momentum�
F
0
1
2
3
4
5
06
17
:8
;9
8
0
1
2
3
04
15
:6
;7
 
�
Dnon_trainable_variables

Elayers
	variables
trainable_variables
Flayer_regularization_losses
Gmetrics
regularization_losses
Hlayer_metrics
 
^\
VARIABLE_VALUElayer_conv2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUElayer_conv2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Inon_trainable_variables

Jlayers
	variables
trainable_variables
Klayer_regularization_losses
Lmetrics
regularization_losses
Mlayer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_61/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_61/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_61/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_61/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�
Nnon_trainable_variables

Olayers
	variables
trainable_variables
Player_regularization_losses
Qmetrics
regularization_losses
Rlayer_metrics
 
 
 
�
Snon_trainable_variables

Tlayers
 	variables
!trainable_variables
Ulayer_regularization_losses
Vmetrics
"regularization_losses
Wlayer_metrics
 
 
 
�
Xnon_trainable_variables

Ylayers
$	variables
%trainable_variables
Zlayer_regularization_losses
[metrics
&regularization_losses
\layer_metrics
 
 
 
�
]non_trainable_variables

^layers
(	variables
)trainable_variables
_layer_regularization_losses
`metrics
*regularization_losses
alayer_metrics
 
 
 
�
bnon_trainable_variables

clayers
,	variables
-trainable_variables
dlayer_regularization_losses
emetrics
.regularization_losses
flayer_metrics
VT
VARIABLE_VALUE
fc1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfc1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
�
gnon_trainable_variables

hlayers
2	variables
3trainable_variables
ilayer_regularization_losses
jmetrics
4regularization_losses
klayer_metrics
 
 
 
�
lnon_trainable_variables

mlayers
6	variables
7trainable_variables
nlayer_regularization_losses
ometrics
8regularization_losses
player_metrics
VT
VARIABLE_VALUE
fc2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEfc2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
�
qnon_trainable_variables

rlayers
<	variables
=trainable_variables
slayer_regularization_losses
tmetrics
>regularization_losses
ulayer_metrics
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE

0
1
F
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
 

v0
w1
 
 
 
 
 
 

0
1
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
4
	xtotal
	ycount
z	variables
{	keras_api
E
	|total
	}count
~
_fn_kwargs
	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

z	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1

	variables
��
VARIABLE_VALUESGD/layer_conv2/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/layer_conv2/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)SGD/batch_normalization_61/gamma/momentumXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/batch_normalization_61/beta/momentumWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/fc1/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/fc1/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/fc2/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/fc2/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_31Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_31layer_conv2/kernellayer_conv2/bias&batch_normalization_61/moving_variancebatch_normalization_61/gamma"batch_normalization_61/moving_meanbatch_normalization_61/beta
fc1/kernelfc1/bias
fc2/kernelfc2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1020014
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&layer_conv2/kernel/Read/ReadVariableOp$layer_conv2/bias/Read/ReadVariableOp0batch_normalization_61/gamma/Read/ReadVariableOp/batch_normalization_61/beta/Read/ReadVariableOp6batch_normalization_61/moving_mean/Read/ReadVariableOp:batch_normalization_61/moving_variance/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3SGD/layer_conv2/kernel/momentum/Read/ReadVariableOp1SGD/layer_conv2/bias/momentum/Read/ReadVariableOp=SGD/batch_normalization_61/gamma/momentum/Read/ReadVariableOp<SGD/batch_normalization_61/beta/momentum/Read/ReadVariableOp+SGD/fc1/kernel/momentum/Read/ReadVariableOp)SGD/fc1/bias/momentum/Read/ReadVariableOp+SGD/fc2/kernel/momentum/Read/ReadVariableOp)SGD/fc2/bias/momentum/Read/ReadVariableOpConst*'
Tin 
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_1020626
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_conv2/kernellayer_conv2/biasbatch_normalization_61/gammabatch_normalization_61/beta"batch_normalization_61/moving_mean&batch_normalization_61/moving_variance
fc1/kernelfc1/bias
fc2/kernelfc2/biasdecaylearning_ratemomentumSGD/itertotalcounttotal_1count_1SGD/layer_conv2/kernel/momentumSGD/layer_conv2/bias/momentum)SGD/batch_normalization_61/gamma/momentum(SGD/batch_normalization_61/beta/momentumSGD/fc1/kernel/momentumSGD/fc1/bias/momentumSGD/fc2/kernel/momentumSGD/fc2/bias/momentum*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1020714��	
�
�
H__inference_layer_conv2_layer_call_and_return_conditional_losses_1019525

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020296

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
e
G__inference_dropout_91_layer_call_and_return_conditional_losses_1020435

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
f
J__inference_activation_61_layer_call_and_return_conditional_losses_1020394

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������� 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
a
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1019487

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�X
�

"__inference__wrapped_model_1019313
input_31U
?predict_layer_conv2_conv1d_expanddims_1_readvariableop_resource: A
3predict_layer_conv2_biasadd_readvariableop_resource: N
@predict_batch_normalization_61_batchnorm_readvariableop_resource: R
Dpredict_batch_normalization_61_batchnorm_mul_readvariableop_resource: P
Bpredict_batch_normalization_61_batchnorm_readvariableop_1_resource: P
Bpredict_batch_normalization_61_batchnorm_readvariableop_2_resource: =
*predict_fc1_matmul_readvariableop_resource:	�(@9
+predict_fc1_biasadd_readvariableop_resource:@<
*predict_fc2_matmul_readvariableop_resource:@9
+predict_fc2_biasadd_readvariableop_resource:
identity��7Predict/batch_normalization_61/batchnorm/ReadVariableOp�9Predict/batch_normalization_61/batchnorm/ReadVariableOp_1�9Predict/batch_normalization_61/batchnorm/ReadVariableOp_2�;Predict/batch_normalization_61/batchnorm/mul/ReadVariableOp�"Predict/fc1/BiasAdd/ReadVariableOp�!Predict/fc1/MatMul/ReadVariableOp�"Predict/fc2/BiasAdd/ReadVariableOp�!Predict/fc2/MatMul/ReadVariableOp�*Predict/layer_conv2/BiasAdd/ReadVariableOp�6Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOp�
)Predict/layer_conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2+
)Predict/layer_conv2/conv1d/ExpandDims/dim�
%Predict/layer_conv2/conv1d/ExpandDims
ExpandDimsinput_312Predict/layer_conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2'
%Predict/layer_conv2/conv1d/ExpandDims�
6Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?predict_layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype028
6Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOp�
+Predict/layer_conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+Predict/layer_conv2/conv1d/ExpandDims_1/dim�
'Predict/layer_conv2/conv1d/ExpandDims_1
ExpandDims>Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOp:value:04Predict/layer_conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2)
'Predict/layer_conv2/conv1d/ExpandDims_1�
Predict/layer_conv2/conv1dConv2D.Predict/layer_conv2/conv1d/ExpandDims:output:00Predict/layer_conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
Predict/layer_conv2/conv1d�
"Predict/layer_conv2/conv1d/SqueezeSqueeze#Predict/layer_conv2/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2$
"Predict/layer_conv2/conv1d/Squeeze�
*Predict/layer_conv2/BiasAdd/ReadVariableOpReadVariableOp3predict_layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*Predict/layer_conv2/BiasAdd/ReadVariableOp�
Predict/layer_conv2/BiasAddBiasAdd+Predict/layer_conv2/conv1d/Squeeze:output:02Predict/layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
Predict/layer_conv2/BiasAdd�
7Predict/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp@predict_batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype029
7Predict/batch_normalization_61/batchnorm/ReadVariableOp�
.Predict/batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.Predict/batch_normalization_61/batchnorm/add/y�
,Predict/batch_normalization_61/batchnorm/addAddV2?Predict/batch_normalization_61/batchnorm/ReadVariableOp:value:07Predict/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2.
,Predict/batch_normalization_61/batchnorm/add�
.Predict/batch_normalization_61/batchnorm/RsqrtRsqrt0Predict/batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes
: 20
.Predict/batch_normalization_61/batchnorm/Rsqrt�
;Predict/batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOpDpredict_batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02=
;Predict/batch_normalization_61/batchnorm/mul/ReadVariableOp�
,Predict/batch_normalization_61/batchnorm/mulMul2Predict/batch_normalization_61/batchnorm/Rsqrt:y:0CPredict/batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,Predict/batch_normalization_61/batchnorm/mul�
.Predict/batch_normalization_61/batchnorm/mul_1Mul$Predict/layer_conv2/BiasAdd:output:00Predict/batch_normalization_61/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 20
.Predict/batch_normalization_61/batchnorm/mul_1�
9Predict/batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOpBpredict_batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02;
9Predict/batch_normalization_61/batchnorm/ReadVariableOp_1�
.Predict/batch_normalization_61/batchnorm/mul_2MulAPredict/batch_normalization_61/batchnorm/ReadVariableOp_1:value:00Predict/batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes
: 20
.Predict/batch_normalization_61/batchnorm/mul_2�
9Predict/batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOpBpredict_batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02;
9Predict/batch_normalization_61/batchnorm/ReadVariableOp_2�
,Predict/batch_normalization_61/batchnorm/subSubAPredict/batch_normalization_61/batchnorm/ReadVariableOp_2:value:02Predict/batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2.
,Predict/batch_normalization_61/batchnorm/sub�
.Predict/batch_normalization_61/batchnorm/add_1AddV22Predict/batch_normalization_61/batchnorm/mul_1:z:00Predict/batch_normalization_61/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 20
.Predict/batch_normalization_61/batchnorm/add_1�
Predict/activation_61/ReluRelu2Predict/batch_normalization_61/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
Predict/activation_61/Relu�
Predict/MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
Predict/MaxPool2/ExpandDims/dim�
Predict/MaxPool2/ExpandDims
ExpandDims(Predict/activation_61/Relu:activations:0(Predict/MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
Predict/MaxPool2/ExpandDims�
Predict/MaxPool2/MaxPoolMaxPool$Predict/MaxPool2/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingSAME*
strides
2
Predict/MaxPool2/MaxPool�
Predict/MaxPool2/SqueezeSqueeze!Predict/MaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2
Predict/MaxPool2/Squeeze�
Predict/dropout_91/IdentityIdentity!Predict/MaxPool2/Squeeze:output:0*
T0*,
_output_shapes
:���������� 2
Predict/dropout_91/Identity�
Predict/flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Predict/flatten_30/Const�
Predict/flatten_30/ReshapeReshape$Predict/dropout_91/Identity:output:0!Predict/flatten_30/Const:output:0*
T0*(
_output_shapes
:����������(2
Predict/flatten_30/Reshape�
!Predict/fc1/MatMul/ReadVariableOpReadVariableOp*predict_fc1_matmul_readvariableop_resource*
_output_shapes
:	�(@*
dtype02#
!Predict/fc1/MatMul/ReadVariableOp�
Predict/fc1/MatMulMatMul#Predict/flatten_30/Reshape:output:0)Predict/fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Predict/fc1/MatMul�
"Predict/fc1/BiasAdd/ReadVariableOpReadVariableOp+predict_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"Predict/fc1/BiasAdd/ReadVariableOp�
Predict/fc1/BiasAddBiasAddPredict/fc1/MatMul:product:0*Predict/fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Predict/fc1/BiasAdd|
Predict/fc1/ReluReluPredict/fc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Predict/fc1/Relu�
Predict/dropout_92/IdentityIdentityPredict/fc1/Relu:activations:0*
T0*'
_output_shapes
:���������@2
Predict/dropout_92/Identity�
!Predict/fc2/MatMul/ReadVariableOpReadVariableOp*predict_fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!Predict/fc2/MatMul/ReadVariableOp�
Predict/fc2/MatMulMatMul$Predict/dropout_92/Identity:output:0)Predict/fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Predict/fc2/MatMul�
"Predict/fc2/BiasAdd/ReadVariableOpReadVariableOp+predict_fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"Predict/fc2/BiasAdd/ReadVariableOp�
Predict/fc2/BiasAddBiasAddPredict/fc2/MatMul:product:0*Predict/fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Predict/fc2/BiasAdd�
Predict/fc2/SoftmaxSoftmaxPredict/fc2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
Predict/fc2/Softmaxx
IdentityIdentityPredict/fc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp8^Predict/batch_normalization_61/batchnorm/ReadVariableOp:^Predict/batch_normalization_61/batchnorm/ReadVariableOp_1:^Predict/batch_normalization_61/batchnorm/ReadVariableOp_2<^Predict/batch_normalization_61/batchnorm/mul/ReadVariableOp#^Predict/fc1/BiasAdd/ReadVariableOp"^Predict/fc1/MatMul/ReadVariableOp#^Predict/fc2/BiasAdd/ReadVariableOp"^Predict/fc2/MatMul/ReadVariableOp+^Predict/layer_conv2/BiasAdd/ReadVariableOp7^Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2r
7Predict/batch_normalization_61/batchnorm/ReadVariableOp7Predict/batch_normalization_61/batchnorm/ReadVariableOp2v
9Predict/batch_normalization_61/batchnorm/ReadVariableOp_19Predict/batch_normalization_61/batchnorm/ReadVariableOp_12v
9Predict/batch_normalization_61/batchnorm/ReadVariableOp_29Predict/batch_normalization_61/batchnorm/ReadVariableOp_22z
;Predict/batch_normalization_61/batchnorm/mul/ReadVariableOp;Predict/batch_normalization_61/batchnorm/mul/ReadVariableOp2H
"Predict/fc1/BiasAdd/ReadVariableOp"Predict/fc1/BiasAdd/ReadVariableOp2F
!Predict/fc1/MatMul/ReadVariableOp!Predict/fc1/MatMul/ReadVariableOp2H
"Predict/fc2/BiasAdd/ReadVariableOp"Predict/fc2/BiasAdd/ReadVariableOp2F
!Predict/fc2/MatMul/ReadVariableOp!Predict/fc2/MatMul/ReadVariableOp2X
*Predict/layer_conv2/BiasAdd/ReadVariableOp*Predict/layer_conv2/BiasAdd/ReadVariableOp2p
6Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOp6Predict/layer_conv2/conv1d/ExpandDims_1/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
input_31
�

�
)__inference_Predict_layer_call_fn_1020039

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	�(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Predict_layer_call_and_return_conditional_losses_10196332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_fc1_layer_call_and_return_conditional_losses_1019602

inputs1
matmul_readvariableop_resource:	�(@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�(@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������(
 
_user_specified_nameinputs
�
e
,__inference_dropout_91_layer_call_fn_1020430

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_10197252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�,
�
D__inference_Predict_layer_call_and_return_conditional_losses_1019869

inputs)
layer_conv2_1019839: !
layer_conv2_1019841: ,
batch_normalization_61_1019844: ,
batch_normalization_61_1019846: ,
batch_normalization_61_1019848: ,
batch_normalization_61_1019850: 
fc1_1019857:	�(@
fc1_1019859:@
fc2_1019863:@
fc2_1019865:
identity��.batch_normalization_61/StatefulPartitionedCall�"dropout_91/StatefulPartitionedCall�"dropout_92/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc2/StatefulPartitionedCall�#layer_conv2/StatefulPartitionedCall�
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_conv2_1019839layer_conv2_1019841*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_layer_conv2_layer_call_and_return_conditional_losses_10195252%
#layer_conv2/StatefulPartitionedCall�
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_61_1019844batch_normalization_61_1019846batch_normalization_61_1019848batch_normalization_61_1019850*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_101978820
.batch_normalization_61/StatefulPartitionedCall�
activation_61/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_10195652
activation_61/PartitionedCall�
MaxPool2/PartitionedCallPartitionedCall&activation_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_MaxPool2_layer_call_and_return_conditional_losses_10195742
MaxPool2/PartitionedCall�
"dropout_91/StatefulPartitionedCallStatefulPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_10197252$
"dropout_91/StatefulPartitionedCall�
flatten_30/PartitionedCallPartitionedCall+dropout_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_flatten_30_layer_call_and_return_conditional_losses_10195892
flatten_30/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0fc1_1019857fc1_1019859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_10196022
fc1/StatefulPartitionedCall�
"dropout_92/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_91/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_10196862$
"dropout_92/StatefulPartitionedCall�
fc2/StatefulPartitionedCallStatefulPartitionedCall+dropout_92/StatefulPartitionedCall:output:0fc2_1019863fc2_1019865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_10196262
fc2/StatefulPartitionedCall
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_61/StatefulPartitionedCall#^dropout_91/StatefulPartitionedCall#^dropout_92/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2H
"dropout_91/StatefulPartitionedCall"dropout_91/StatefulPartitionedCall2H
"dropout_92/StatefulPartitionedCall"dropout_92/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
Ӏ
�	
D__inference_Predict_layer_call_and_return_conditional_losses_1020200

inputsM
7layer_conv2_conv1d_expanddims_1_readvariableop_resource: 9
+layer_conv2_biasadd_readvariableop_resource: L
>batch_normalization_61_assignmovingavg_readvariableop_resource: N
@batch_normalization_61_assignmovingavg_1_readvariableop_resource: J
<batch_normalization_61_batchnorm_mul_readvariableop_resource: F
8batch_normalization_61_batchnorm_readvariableop_resource: 5
"fc1_matmul_readvariableop_resource:	�(@1
#fc1_biasadd_readvariableop_resource:@4
"fc2_matmul_readvariableop_resource:@1
#fc2_biasadd_readvariableop_resource:
identity��&batch_normalization_61/AssignMovingAvg�5batch_normalization_61/AssignMovingAvg/ReadVariableOp�(batch_normalization_61/AssignMovingAvg_1�7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp�/batch_normalization_61/batchnorm/ReadVariableOp�3batch_normalization_61/batchnorm/mul/ReadVariableOp�fc1/BiasAdd/ReadVariableOp�fc1/MatMul/ReadVariableOp�fc2/BiasAdd/ReadVariableOp�fc2/MatMul/ReadVariableOp�"layer_conv2/BiasAdd/ReadVariableOp�.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp�
!layer_conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!layer_conv2/conv1d/ExpandDims/dim�
layer_conv2/conv1d/ExpandDims
ExpandDimsinputs*layer_conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
layer_conv2/conv1d/ExpandDims�
.layer_conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp7layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype020
.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp�
#layer_conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#layer_conv2/conv1d/ExpandDims_1/dim�
layer_conv2/conv1d/ExpandDims_1
ExpandDims6layer_conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0,layer_conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2!
layer_conv2/conv1d/ExpandDims_1�
layer_conv2/conv1dConv2D&layer_conv2/conv1d/ExpandDims:output:0(layer_conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
layer_conv2/conv1d�
layer_conv2/conv1d/SqueezeSqueezelayer_conv2/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
layer_conv2/conv1d/Squeeze�
"layer_conv2/BiasAdd/ReadVariableOpReadVariableOp+layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"layer_conv2/BiasAdd/ReadVariableOp�
layer_conv2/BiasAddBiasAdd#layer_conv2/conv1d/Squeeze:output:0*layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
layer_conv2/BiasAdd�
5batch_normalization_61/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       27
5batch_normalization_61/moments/mean/reduction_indices�
#batch_normalization_61/moments/meanMeanlayer_conv2/BiasAdd:output:0>batch_normalization_61/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2%
#batch_normalization_61/moments/mean�
+batch_normalization_61/moments/StopGradientStopGradient,batch_normalization_61/moments/mean:output:0*
T0*"
_output_shapes
: 2-
+batch_normalization_61/moments/StopGradient�
0batch_normalization_61/moments/SquaredDifferenceSquaredDifferencelayer_conv2/BiasAdd:output:04batch_normalization_61/moments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 22
0batch_normalization_61/moments/SquaredDifference�
9batch_normalization_61/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2;
9batch_normalization_61/moments/variance/reduction_indices�
'batch_normalization_61/moments/varianceMean4batch_normalization_61/moments/SquaredDifference:z:0Bbatch_normalization_61/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2)
'batch_normalization_61/moments/variance�
&batch_normalization_61/moments/SqueezeSqueeze,batch_normalization_61/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2(
&batch_normalization_61/moments/Squeeze�
(batch_normalization_61/moments/Squeeze_1Squeeze0batch_normalization_61/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2*
(batch_normalization_61/moments/Squeeze_1�
,batch_normalization_61/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_61/AssignMovingAvg/decay�
5batch_normalization_61/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_61_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_61/AssignMovingAvg/ReadVariableOp�
*batch_normalization_61/AssignMovingAvg/subSub=batch_normalization_61/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_61/moments/Squeeze:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_61/AssignMovingAvg/sub�
*batch_normalization_61/AssignMovingAvg/mulMul.batch_normalization_61/AssignMovingAvg/sub:z:05batch_normalization_61/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2,
*batch_normalization_61/AssignMovingAvg/mul�
&batch_normalization_61/AssignMovingAvgAssignSubVariableOp>batch_normalization_61_assignmovingavg_readvariableop_resource.batch_normalization_61/AssignMovingAvg/mul:z:06^batch_normalization_61/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02(
&batch_normalization_61/AssignMovingAvg�
.batch_normalization_61/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_61/AssignMovingAvg_1/decay�
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_61_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype029
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_61/AssignMovingAvg_1/subSub?batch_normalization_61/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_61/moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_61/AssignMovingAvg_1/sub�
,batch_normalization_61/AssignMovingAvg_1/mulMul0batch_normalization_61/AssignMovingAvg_1/sub:z:07batch_normalization_61/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2.
,batch_normalization_61/AssignMovingAvg_1/mul�
(batch_normalization_61/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_61_assignmovingavg_1_readvariableop_resource0batch_normalization_61/AssignMovingAvg_1/mul:z:08^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02*
(batch_normalization_61/AssignMovingAvg_1�
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_61/batchnorm/add/y�
$batch_normalization_61/batchnorm/addAddV21batch_normalization_61/moments/Squeeze_1:output:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_61/batchnorm/add�
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_61/batchnorm/Rsqrt�
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOp�
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_61/batchnorm/mul�
&batch_normalization_61/batchnorm/mul_1Mullayer_conv2/BiasAdd:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2(
&batch_normalization_61/batchnorm/mul_1�
&batch_normalization_61/batchnorm/mul_2Mul/batch_normalization_61/moments/Squeeze:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_61/batchnorm/mul_2�
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_61/batchnorm/ReadVariableOp�
$batch_normalization_61/batchnorm/subSub7batch_normalization_61/batchnorm/ReadVariableOp:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_61/batchnorm/sub�
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2(
&batch_normalization_61/batchnorm/add_1�
activation_61/ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation_61/Relut
MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPool2/ExpandDims/dim�
MaxPool2/ExpandDims
ExpandDims activation_61/Relu:activations:0 MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
MaxPool2/ExpandDims�
MaxPool2/MaxPoolMaxPoolMaxPool2/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingSAME*
strides
2
MaxPool2/MaxPool�
MaxPool2/SqueezeSqueezeMaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2
MaxPool2/Squeezey
dropout_91/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
dropout_91/dropout/Const�
dropout_91/dropout/MulMulMaxPool2/Squeeze:output:0!dropout_91/dropout/Const:output:0*
T0*,
_output_shapes
:���������� 2
dropout_91/dropout/Mul}
dropout_91/dropout/ShapeShapeMaxPool2/Squeeze:output:0*
T0*
_output_shapes
:2
dropout_91/dropout/Shape�
/dropout_91/dropout/random_uniform/RandomUniformRandomUniform!dropout_91/dropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
dtype021
/dropout_91/dropout/random_uniform/RandomUniform�
!dropout_91/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>2#
!dropout_91/dropout/GreaterEqual/y�
dropout_91/dropout/GreaterEqualGreaterEqual8dropout_91/dropout/random_uniform/RandomUniform:output:0*dropout_91/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������� 2!
dropout_91/dropout/GreaterEqual�
dropout_91/dropout/CastCast#dropout_91/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout_91/dropout/Cast�
dropout_91/dropout/Mul_1Muldropout_91/dropout/Mul:z:0dropout_91/dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout_91/dropout/Mul_1u
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_30/Const�
flatten_30/ReshapeReshapedropout_91/dropout/Mul_1:z:0flatten_30/Const:output:0*
T0*(
_output_shapes
:����������(2
flatten_30/Reshape�
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes
:	�(@*
dtype02
fc1/MatMul/ReadVariableOp�

fc1/MatMulMatMulflatten_30/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

fc1/MatMul�
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc1/BiasAdd/ReadVariableOp�
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
fc1/BiasAddd
fc1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2

fc1/Reluy
dropout_92/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
dropout_92/dropout/Const�
dropout_92/dropout/MulMulfc1/Relu:activations:0!dropout_92/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_92/dropout/Mulz
dropout_92/dropout/ShapeShapefc1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_92/dropout/Shape�
/dropout_92/dropout/random_uniform/RandomUniformRandomUniform!dropout_92/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype021
/dropout_92/dropout/random_uniform/RandomUniform�
!dropout_92/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>2#
!dropout_92/dropout/GreaterEqual/y�
dropout_92/dropout/GreaterEqualGreaterEqual8dropout_92/dropout/random_uniform/RandomUniform:output:0*dropout_92/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_92/dropout/GreaterEqual�
dropout_92/dropout/CastCast#dropout_92/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_92/dropout/Cast�
dropout_92/dropout/Mul_1Muldropout_92/dropout/Mul:z:0dropout_92/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_92/dropout/Mul_1�
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
fc2/MatMul/ReadVariableOp�

fc2/MatMulMatMuldropout_92/dropout/Mul_1:z:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

fc2/MatMul�
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc2/BiasAdd/ReadVariableOp�
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
fc2/BiasAddm
fc2/SoftmaxSoftmaxfc2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
fc2/Softmaxp
IdentityIdentityfc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp'^batch_normalization_61/AssignMovingAvg6^batch_normalization_61/AssignMovingAvg/ReadVariableOp)^batch_normalization_61/AssignMovingAvg_18^batch_normalization_61/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_61/batchnorm/ReadVariableOp4^batch_normalization_61/batchnorm/mul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^layer_conv2/BiasAdd/ReadVariableOp/^layer_conv2/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2P
&batch_normalization_61/AssignMovingAvg&batch_normalization_61/AssignMovingAvg2n
5batch_normalization_61/AssignMovingAvg/ReadVariableOp5batch_normalization_61/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_61/AssignMovingAvg_1(batch_normalization_61/AssignMovingAvg_12r
7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp7batch_normalization_61/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp2H
"layer_conv2/BiasAdd/ReadVariableOp"layer_conv2/BiasAdd/ReadVariableOp2`
.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�+
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020384

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020350

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
F
*__inference_MaxPool2_layer_call_fn_1020404

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_MaxPool2_layer_call_and_return_conditional_losses_10195742
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
f
G__inference_dropout_91_layer_call_and_return_conditional_losses_1020447

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
f
G__inference_dropout_91_layer_call_and_return_conditional_losses_1019725

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:���������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:���������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:���������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:���������� 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:���������� 2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
a
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1020420

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingSAME*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�)
�
D__inference_Predict_layer_call_and_return_conditional_losses_1019950
input_31)
layer_conv2_1019920: !
layer_conv2_1019922: ,
batch_normalization_61_1019925: ,
batch_normalization_61_1019927: ,
batch_normalization_61_1019929: ,
batch_normalization_61_1019931: 
fc1_1019938:	�(@
fc1_1019940:@
fc2_1019944:@
fc2_1019946:
identity��.batch_normalization_61/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc2/StatefulPartitionedCall�#layer_conv2/StatefulPartitionedCall�
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_31layer_conv2_1019920layer_conv2_1019922*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_layer_conv2_layer_call_and_return_conditional_losses_10195252%
#layer_conv2/StatefulPartitionedCall�
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_61_1019925batch_normalization_61_1019927batch_normalization_61_1019929batch_normalization_61_1019931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_101955020
.batch_normalization_61/StatefulPartitionedCall�
activation_61/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_10195652
activation_61/PartitionedCall�
MaxPool2/PartitionedCallPartitionedCall&activation_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_MaxPool2_layer_call_and_return_conditional_losses_10195742
MaxPool2/PartitionedCall�
dropout_91/PartitionedCallPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_10195812
dropout_91/PartitionedCall�
flatten_30/PartitionedCallPartitionedCall#dropout_91/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_flatten_30_layer_call_and_return_conditional_losses_10195892
flatten_30/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0fc1_1019938fc1_1019940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_10196022
fc1/StatefulPartitionedCall�
dropout_92/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_10196132
dropout_92/PartitionedCall�
fc2/StatefulPartitionedCallStatefulPartitionedCall#dropout_92/PartitionedCall:output:0fc2_1019944fc2_1019946*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_10196262
fc2/StatefulPartitionedCall
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_61/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
input_31
�

�
%__inference_signature_wrapper_1020014
input_31
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	�(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_10193132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
input_31
�
c
G__inference_flatten_30_layer_call_and_return_conditional_losses_1019589

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
e
G__inference_dropout_92_layer_call_and_return_conditional_losses_1019613

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
@__inference_fc1_layer_call_and_return_conditional_losses_1020478

inputs1
matmul_readvariableop_resource:	�(@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�(@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������(
 
_user_specified_nameinputs
�
H
,__inference_flatten_30_layer_call_fn_1020452

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_flatten_30_layer_call_and_return_conditional_losses_10195892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
a
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1020412

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
e
G__inference_dropout_92_layer_call_and_return_conditional_losses_1020493

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1019337

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
f
G__inference_dropout_92_layer_call_and_return_conditional_losses_1019686

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�r
�
#__inference__traced_restore_1020714
file_prefix9
#assignvariableop_layer_conv2_kernel: 1
#assignvariableop_1_layer_conv2_bias: =
/assignvariableop_2_batch_normalization_61_gamma: <
.assignvariableop_3_batch_normalization_61_beta: C
5assignvariableop_4_batch_normalization_61_moving_mean: G
9assignvariableop_5_batch_normalization_61_moving_variance: 0
assignvariableop_6_fc1_kernel:	�(@)
assignvariableop_7_fc1_bias:@/
assignvariableop_8_fc2_kernel:@)
assignvariableop_9_fc2_bias:#
assignvariableop_10_decay: +
!assignvariableop_11_learning_rate: &
assignvariableop_12_momentum: &
assignvariableop_13_sgd_iter:	 #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: I
3assignvariableop_18_sgd_layer_conv2_kernel_momentum: ?
1assignvariableop_19_sgd_layer_conv2_bias_momentum: K
=assignvariableop_20_sgd_batch_normalization_61_gamma_momentum: J
<assignvariableop_21_sgd_batch_normalization_61_beta_momentum: >
+assignvariableop_22_sgd_fc1_kernel_momentum:	�(@7
)assignvariableop_23_sgd_fc1_bias_momentum:@=
+assignvariableop_24_sgd_fc2_kernel_momentum:@7
)assignvariableop_25_sgd_fc2_bias_momentum:
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp#assignvariableop_layer_conv2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_layer_conv2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_61_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_61_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_61_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_61_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_fc2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_fc2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_momentumIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_sgd_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_sgd_layer_conv2_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp1assignvariableop_19_sgd_layer_conv2_bias_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp=assignvariableop_20_sgd_batch_normalization_61_gamma_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp<assignvariableop_21_sgd_batch_normalization_61_beta_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_sgd_fc1_kernel_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_sgd_fc1_bias_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_sgd_fc2_kernel_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_sgd_fc2_bias_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_259
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_26f
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_27�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
%__inference_fc1_layer_call_fn_1020467

inputs
unknown:	�(@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_10196022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������(
 
_user_specified_nameinputs
�
e
G__inference_dropout_91_layer_call_and_return_conditional_losses_1019581

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:���������� 2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:���������� 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_61_layer_call_fn_1020263

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_10195502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�)
�
D__inference_Predict_layer_call_and_return_conditional_losses_1019633

inputs)
layer_conv2_1019526: !
layer_conv2_1019528: ,
batch_normalization_61_1019551: ,
batch_normalization_61_1019553: ,
batch_normalization_61_1019555: ,
batch_normalization_61_1019557: 
fc1_1019603:	�(@
fc1_1019605:@
fc2_1019627:@
fc2_1019629:
identity��.batch_normalization_61/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc2/StatefulPartitionedCall�#layer_conv2/StatefulPartitionedCall�
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_conv2_1019526layer_conv2_1019528*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_layer_conv2_layer_call_and_return_conditional_losses_10195252%
#layer_conv2/StatefulPartitionedCall�
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_61_1019551batch_normalization_61_1019553batch_normalization_61_1019555batch_normalization_61_1019557*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_101955020
.batch_normalization_61/StatefulPartitionedCall�
activation_61/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_10195652
activation_61/PartitionedCall�
MaxPool2/PartitionedCallPartitionedCall&activation_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_MaxPool2_layer_call_and_return_conditional_losses_10195742
MaxPool2/PartitionedCall�
dropout_91/PartitionedCallPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_10195812
dropout_91/PartitionedCall�
flatten_30/PartitionedCallPartitionedCall#dropout_91/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_flatten_30_layer_call_and_return_conditional_losses_10195892
flatten_30/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0fc1_1019603fc1_1019605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_10196022
fc1/StatefulPartitionedCall�
dropout_92/PartitionedCallPartitionedCall$fc1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_10196132
dropout_92/PartitionedCall�
fc2/StatefulPartitionedCallStatefulPartitionedCall#dropout_92/PartitionedCall:output:0fc2_1019627fc2_1019629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_10196262
fc2/StatefulPartitionedCall
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_61/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�M
�	
D__inference_Predict_layer_call_and_return_conditional_losses_1020118

inputsM
7layer_conv2_conv1d_expanddims_1_readvariableop_resource: 9
+layer_conv2_biasadd_readvariableop_resource: F
8batch_normalization_61_batchnorm_readvariableop_resource: J
<batch_normalization_61_batchnorm_mul_readvariableop_resource: H
:batch_normalization_61_batchnorm_readvariableop_1_resource: H
:batch_normalization_61_batchnorm_readvariableop_2_resource: 5
"fc1_matmul_readvariableop_resource:	�(@1
#fc1_biasadd_readvariableop_resource:@4
"fc2_matmul_readvariableop_resource:@1
#fc2_biasadd_readvariableop_resource:
identity��/batch_normalization_61/batchnorm/ReadVariableOp�1batch_normalization_61/batchnorm/ReadVariableOp_1�1batch_normalization_61/batchnorm/ReadVariableOp_2�3batch_normalization_61/batchnorm/mul/ReadVariableOp�fc1/BiasAdd/ReadVariableOp�fc1/MatMul/ReadVariableOp�fc2/BiasAdd/ReadVariableOp�fc2/MatMul/ReadVariableOp�"layer_conv2/BiasAdd/ReadVariableOp�.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp�
!layer_conv2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!layer_conv2/conv1d/ExpandDims/dim�
layer_conv2/conv1d/ExpandDims
ExpandDimsinputs*layer_conv2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
layer_conv2/conv1d/ExpandDims�
.layer_conv2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp7layer_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype020
.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp�
#layer_conv2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#layer_conv2/conv1d/ExpandDims_1/dim�
layer_conv2/conv1d/ExpandDims_1
ExpandDims6layer_conv2/conv1d/ExpandDims_1/ReadVariableOp:value:0,layer_conv2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2!
layer_conv2/conv1d/ExpandDims_1�
layer_conv2/conv1dConv2D&layer_conv2/conv1d/ExpandDims:output:0(layer_conv2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
layer_conv2/conv1d�
layer_conv2/conv1d/SqueezeSqueezelayer_conv2/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
layer_conv2/conv1d/Squeeze�
"layer_conv2/BiasAdd/ReadVariableOpReadVariableOp+layer_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"layer_conv2/BiasAdd/ReadVariableOp�
layer_conv2/BiasAddBiasAdd#layer_conv2/conv1d/Squeeze:output:0*layer_conv2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
layer_conv2/BiasAdd�
/batch_normalization_61/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_61_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/batch_normalization_61/batchnorm/ReadVariableOp�
&batch_normalization_61/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_61/batchnorm/add/y�
$batch_normalization_61/batchnorm/addAddV27batch_normalization_61/batchnorm/ReadVariableOp:value:0/batch_normalization_61/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2&
$batch_normalization_61/batchnorm/add�
&batch_normalization_61/batchnorm/RsqrtRsqrt(batch_normalization_61/batchnorm/add:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_61/batchnorm/Rsqrt�
3batch_normalization_61/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_61_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization_61/batchnorm/mul/ReadVariableOp�
$batch_normalization_61/batchnorm/mulMul*batch_normalization_61/batchnorm/Rsqrt:y:0;batch_normalization_61/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2&
$batch_normalization_61/batchnorm/mul�
&batch_normalization_61/batchnorm/mul_1Mullayer_conv2/BiasAdd:output:0(batch_normalization_61/batchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2(
&batch_normalization_61/batchnorm/mul_1�
1batch_normalization_61/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_1�
&batch_normalization_61/batchnorm/mul_2Mul9batch_normalization_61/batchnorm/ReadVariableOp_1:value:0(batch_normalization_61/batchnorm/mul:z:0*
T0*
_output_shapes
: 2(
&batch_normalization_61/batchnorm/mul_2�
1batch_normalization_61/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_61_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype023
1batch_normalization_61/batchnorm/ReadVariableOp_2�
$batch_normalization_61/batchnorm/subSub9batch_normalization_61/batchnorm/ReadVariableOp_2:value:0*batch_normalization_61/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2&
$batch_normalization_61/batchnorm/sub�
&batch_normalization_61/batchnorm/add_1AddV2*batch_normalization_61/batchnorm/mul_1:z:0(batch_normalization_61/batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2(
&batch_normalization_61/batchnorm/add_1�
activation_61/ReluRelu*batch_normalization_61/batchnorm/add_1:z:0*
T0*,
_output_shapes
:���������� 2
activation_61/Relut
MaxPool2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPool2/ExpandDims/dim�
MaxPool2/ExpandDims
ExpandDims activation_61/Relu:activations:0 MaxPool2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2
MaxPool2/ExpandDims�
MaxPool2/MaxPoolMaxPoolMaxPool2/ExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingSAME*
strides
2
MaxPool2/MaxPool�
MaxPool2/SqueezeSqueezeMaxPool2/MaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2
MaxPool2/Squeeze�
dropout_91/IdentityIdentityMaxPool2/Squeeze:output:0*
T0*,
_output_shapes
:���������� 2
dropout_91/Identityu
flatten_30/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_30/Const�
flatten_30/ReshapeReshapedropout_91/Identity:output:0flatten_30/Const:output:0*
T0*(
_output_shapes
:����������(2
flatten_30/Reshape�
fc1/MatMul/ReadVariableOpReadVariableOp"fc1_matmul_readvariableop_resource*
_output_shapes
:	�(@*
dtype02
fc1/MatMul/ReadVariableOp�

fc1/MatMulMatMulflatten_30/Reshape:output:0!fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2

fc1/MatMul�
fc1/BiasAdd/ReadVariableOpReadVariableOp#fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
fc1/BiasAdd/ReadVariableOp�
fc1/BiasAddBiasAddfc1/MatMul:product:0"fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
fc1/BiasAddd
fc1/ReluRelufc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2

fc1/Relu�
dropout_92/IdentityIdentityfc1/Relu:activations:0*
T0*'
_output_shapes
:���������@2
dropout_92/Identity�
fc2/MatMul/ReadVariableOpReadVariableOp"fc2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
fc2/MatMul/ReadVariableOp�

fc2/MatMulMatMuldropout_92/Identity:output:0!fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

fc2/MatMul�
fc2/BiasAdd/ReadVariableOpReadVariableOp#fc2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc2/BiasAdd/ReadVariableOp�
fc2/BiasAddBiasAddfc2/MatMul:product:0"fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
fc2/BiasAddm
fc2/SoftmaxSoftmaxfc2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
fc2/Softmaxp
IdentityIdentityfc2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp0^batch_normalization_61/batchnorm/ReadVariableOp2^batch_normalization_61/batchnorm/ReadVariableOp_12^batch_normalization_61/batchnorm/ReadVariableOp_24^batch_normalization_61/batchnorm/mul/ReadVariableOp^fc1/BiasAdd/ReadVariableOp^fc1/MatMul/ReadVariableOp^fc2/BiasAdd/ReadVariableOp^fc2/MatMul/ReadVariableOp#^layer_conv2/BiasAdd/ReadVariableOp/^layer_conv2/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2b
/batch_normalization_61/batchnorm/ReadVariableOp/batch_normalization_61/batchnorm/ReadVariableOp2f
1batch_normalization_61/batchnorm/ReadVariableOp_11batch_normalization_61/batchnorm/ReadVariableOp_12f
1batch_normalization_61/batchnorm/ReadVariableOp_21batch_normalization_61/batchnorm/ReadVariableOp_22j
3batch_normalization_61/batchnorm/mul/ReadVariableOp3batch_normalization_61/batchnorm/mul/ReadVariableOp28
fc1/BiasAdd/ReadVariableOpfc1/BiasAdd/ReadVariableOp26
fc1/MatMul/ReadVariableOpfc1/MatMul/ReadVariableOp28
fc2/BiasAdd/ReadVariableOpfc2/BiasAdd/ReadVariableOp26
fc2/MatMul/ReadVariableOpfc2/MatMul/ReadVariableOp2H
"layer_conv2/BiasAdd/ReadVariableOp"layer_conv2/BiasAdd/ReadVariableOp2`
.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp.layer_conv2/conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_layer_conv2_layer_call_fn_1020209

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_layer_conv2_layer_call_and_return_conditional_losses_10195252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_61_layer_call_fn_1020237

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_10193372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�

�
)__inference_Predict_layer_call_fn_1019917
input_31
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	�(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Predict_layer_call_and_return_conditional_losses_10198692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
input_31
�+
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1019397

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�

�
)__inference_Predict_layer_call_fn_1020064

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	�(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Predict_layer_call_and_return_conditional_losses_10198692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_layer_conv2_layer_call_and_return_conditional_losses_1020224

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_61_layer_call_fn_1020276

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_10197882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
�
@__inference_fc2_layer_call_and_return_conditional_losses_1020525

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
H
,__inference_dropout_92_layer_call_fn_1020483

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_10196132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
@__inference_fc2_layer_call_and_return_conditional_losses_1019626

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
%__inference_fc2_layer_call_fn_1020514

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_10196262
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�+
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1019788

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:���������� 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�

�
)__inference_Predict_layer_call_fn_1019656
input_31
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:	�(@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_31unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Predict_layer_call_and_return_conditional_losses_10196332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
input_31
�
f
G__inference_dropout_92_layer_call_and_return_conditional_losses_1020505

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1019550

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:���������� 2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*,
_output_shapes
:���������� 2

Identity�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������� : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
F
*__inference_MaxPool2_layer_call_fn_1020399

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_MaxPool2_layer_call_and_return_conditional_losses_10194872
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
c
G__inference_flatten_30_layer_call_and_return_conditional_losses_1020458

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�=
�
 __inference__traced_save_1020626
file_prefix1
-savev2_layer_conv2_kernel_read_readvariableop/
+savev2_layer_conv2_bias_read_readvariableop;
7savev2_batch_normalization_61_gamma_read_readvariableop:
6savev2_batch_normalization_61_beta_read_readvariableopA
=savev2_batch_normalization_61_moving_mean_read_readvariableopE
Asavev2_batch_normalization_61_moving_variance_read_readvariableop)
%savev2_fc1_kernel_read_readvariableop'
#savev2_fc1_bias_read_readvariableop)
%savev2_fc2_kernel_read_readvariableop'
#savev2_fc2_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_sgd_layer_conv2_kernel_momentum_read_readvariableop<
8savev2_sgd_layer_conv2_bias_momentum_read_readvariableopH
Dsavev2_sgd_batch_normalization_61_gamma_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_61_beta_momentum_read_readvariableop6
2savev2_sgd_fc1_kernel_momentum_read_readvariableop4
0savev2_sgd_fc1_bias_momentum_read_readvariableop6
2savev2_sgd_fc2_kernel_momentum_read_readvariableop4
0savev2_sgd_fc2_bias_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_layer_conv2_kernel_read_readvariableop+savev2_layer_conv2_bias_read_readvariableop7savev2_batch_normalization_61_gamma_read_readvariableop6savev2_batch_normalization_61_beta_read_readvariableop=savev2_batch_normalization_61_moving_mean_read_readvariableopAsavev2_batch_normalization_61_moving_variance_read_readvariableop%savev2_fc1_kernel_read_readvariableop#savev2_fc1_bias_read_readvariableop%savev2_fc2_kernel_read_readvariableop#savev2_fc2_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_sgd_layer_conv2_kernel_momentum_read_readvariableop8savev2_sgd_layer_conv2_bias_momentum_read_readvariableopDsavev2_sgd_batch_normalization_61_gamma_momentum_read_readvariableopCsavev2_sgd_batch_normalization_61_beta_momentum_read_readvariableop2savev2_sgd_fc1_kernel_momentum_read_readvariableop0savev2_sgd_fc1_bias_momentum_read_readvariableop2savev2_sgd_fc2_kernel_momentum_read_readvariableop0savev2_sgd_fc2_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : :	�(@:@:@:: : : : : : : : : : : : :	�(@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	�(@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	�(@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�+
�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020330

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg/mul�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 2
AssignMovingAvg_1/mul�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identity�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_61_layer_call_fn_1020250

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_10193972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
a
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1019574

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*0
_output_shapes
:���������� *
ksize
*
paddingSAME*
strides
2	
MaxPool}
SqueezeSqueezeMaxPool:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims
2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
K
/__inference_activation_61_layer_call_fn_1020389

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_10195652
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
f
J__inference_activation_61_layer_call_and_return_conditional_losses_1019565

inputs
identityS
ReluReluinputs*
T0*,
_output_shapes
:���������� 2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�,
�
D__inference_Predict_layer_call_and_return_conditional_losses_1019983
input_31)
layer_conv2_1019953: !
layer_conv2_1019955: ,
batch_normalization_61_1019958: ,
batch_normalization_61_1019960: ,
batch_normalization_61_1019962: ,
batch_normalization_61_1019964: 
fc1_1019971:	�(@
fc1_1019973:@
fc2_1019977:@
fc2_1019979:
identity��.batch_normalization_61/StatefulPartitionedCall�"dropout_91/StatefulPartitionedCall�"dropout_92/StatefulPartitionedCall�fc1/StatefulPartitionedCall�fc2/StatefulPartitionedCall�#layer_conv2/StatefulPartitionedCall�
#layer_conv2/StatefulPartitionedCallStatefulPartitionedCallinput_31layer_conv2_1019953layer_conv2_1019955*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_layer_conv2_layer_call_and_return_conditional_losses_10195252%
#layer_conv2/StatefulPartitionedCall�
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall,layer_conv2/StatefulPartitionedCall:output:0batch_normalization_61_1019958batch_normalization_61_1019960batch_normalization_61_1019962batch_normalization_61_1019964*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_101978820
.batch_normalization_61/StatefulPartitionedCall�
activation_61/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_61_layer_call_and_return_conditional_losses_10195652
activation_61/PartitionedCall�
MaxPool2/PartitionedCallPartitionedCall&activation_61/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_MaxPool2_layer_call_and_return_conditional_losses_10195742
MaxPool2/PartitionedCall�
"dropout_91/StatefulPartitionedCallStatefulPartitionedCall!MaxPool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_10197252$
"dropout_91/StatefulPartitionedCall�
flatten_30/PartitionedCallPartitionedCall+dropout_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_flatten_30_layer_call_and_return_conditional_losses_10195892
flatten_30/PartitionedCall�
fc1/StatefulPartitionedCallStatefulPartitionedCall#flatten_30/PartitionedCall:output:0fc1_1019971fc1_1019973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc1_layer_call_and_return_conditional_losses_10196022
fc1/StatefulPartitionedCall�
"dropout_92/StatefulPartitionedCallStatefulPartitionedCall$fc1/StatefulPartitionedCall:output:0#^dropout_91/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_10196862$
"dropout_92/StatefulPartitionedCall�
fc2/StatefulPartitionedCallStatefulPartitionedCall+dropout_92/StatefulPartitionedCall:output:0fc2_1019977fc2_1019979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_fc2_layer_call_and_return_conditional_losses_10196262
fc2/StatefulPartitionedCall
IdentityIdentity$fc2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^batch_normalization_61/StatefulPartitionedCall#^dropout_91/StatefulPartitionedCall#^dropout_92/StatefulPartitionedCall^fc1/StatefulPartitionedCall^fc2/StatefulPartitionedCall$^layer_conv2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : 2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2H
"dropout_91/StatefulPartitionedCall"dropout_91/StatefulPartitionedCall2H
"dropout_92/StatefulPartitionedCall"dropout_92/StatefulPartitionedCall2:
fc1/StatefulPartitionedCallfc1/StatefulPartitionedCall2:
fc2/StatefulPartitionedCallfc2/StatefulPartitionedCall2J
#layer_conv2/StatefulPartitionedCall#layer_conv2/StatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
input_31
�
H
,__inference_dropout_91_layer_call_fn_1020425

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_91_layer_call_and_return_conditional_losses_10195812
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������� :T P
,
_output_shapes
:���������� 
 
_user_specified_nameinputs
�
e
,__inference_dropout_92_layer_call_fn_1020488

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dropout_92_layer_call_and_return_conditional_losses_10196862
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
B
input_316
serving_default_input_31:0����������7
fc20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer_with_weights-3

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	@decay
Alearning_rate
Bmomentum
Citermomentum�momentum�momentum�momentum�0momentum�1momentum�:momentum�;momentum�"
	optimizer
f
0
1
2
3
4
5
06
17
:8
;9"
trackable_list_wrapper
X
0
1
2
3
04
15
:6
;7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dnon_trainable_variables

Elayers
	variables
trainable_variables
Flayer_regularization_losses
Gmetrics
regularization_losses
Hlayer_metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
(:& 2layer_conv2/kernel
: 2layer_conv2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
	variables
trainable_variables
Klayer_regularization_losses
Lmetrics
regularization_losses
Mlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_61/gamma
):' 2batch_normalization_61/beta
2:0  (2"batch_normalization_61/moving_mean
6:4  (2&batch_normalization_61/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
	variables
trainable_variables
Player_regularization_losses
Qmetrics
regularization_losses
Rlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
 	variables
!trainable_variables
Ulayer_regularization_losses
Vmetrics
"regularization_losses
Wlayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
$	variables
%trainable_variables
Zlayer_regularization_losses
[metrics
&regularization_losses
\layer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
(	variables
)trainable_variables
_layer_regularization_losses
`metrics
*regularization_losses
alayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
bnon_trainable_variables

clayers
,	variables
-trainable_variables
dlayer_regularization_losses
emetrics
.regularization_losses
flayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�(@2
fc1/kernel
:@2fc1/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
2	variables
3trainable_variables
ilayer_regularization_losses
jmetrics
4regularization_losses
klayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
lnon_trainable_variables

mlayers
6	variables
7trainable_variables
nlayer_regularization_losses
ometrics
8regularization_losses
player_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:@2
fc2/kernel
:2fc2/bias
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
�
qnon_trainable_variables

rlayers
<	variables
=trainable_variables
slayer_regularization_losses
tmetrics
>regularization_losses
ulayer_metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
.
0
1"
trackable_list_wrapper
f
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
9"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
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
.
0
1"
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
N
	xtotal
	ycount
z	variables
{	keras_api"
_tf_keras_metric
_
	|total
	}count
~
_fn_kwargs
	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
-
	variables"
_generic_user_object
3:1 2SGD/layer_conv2/kernel/momentum
):' 2SGD/layer_conv2/bias/momentum
5:3 2)SGD/batch_normalization_61/gamma/momentum
4:2 2(SGD/batch_normalization_61/beta/momentum
(:&	�(@2SGD/fc1/kernel/momentum
!:@2SGD/fc1/bias/momentum
':%@2SGD/fc2/kernel/momentum
!:2SGD/fc2/bias/momentum
�2�
)__inference_Predict_layer_call_fn_1019656
)__inference_Predict_layer_call_fn_1020039
)__inference_Predict_layer_call_fn_1020064
)__inference_Predict_layer_call_fn_1019917�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_Predict_layer_call_and_return_conditional_losses_1020118
D__inference_Predict_layer_call_and_return_conditional_losses_1020200
D__inference_Predict_layer_call_and_return_conditional_losses_1019950
D__inference_Predict_layer_call_and_return_conditional_losses_1019983�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_1019313input_31"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_layer_conv2_layer_call_fn_1020209�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_layer_conv2_layer_call_and_return_conditional_losses_1020224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_61_layer_call_fn_1020237
8__inference_batch_normalization_61_layer_call_fn_1020250
8__inference_batch_normalization_61_layer_call_fn_1020263
8__inference_batch_normalization_61_layer_call_fn_1020276�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020296
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020330
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020350
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020384�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_activation_61_layer_call_fn_1020389�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_activation_61_layer_call_and_return_conditional_losses_1020394�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_MaxPool2_layer_call_fn_1020399
*__inference_MaxPool2_layer_call_fn_1020404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1020412
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1020420�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_dropout_91_layer_call_fn_1020425
,__inference_dropout_91_layer_call_fn_1020430�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_91_layer_call_and_return_conditional_losses_1020435
G__inference_dropout_91_layer_call_and_return_conditional_losses_1020447�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_flatten_30_layer_call_fn_1020452�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_flatten_30_layer_call_and_return_conditional_losses_1020458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_fc1_layer_call_fn_1020467�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_fc1_layer_call_and_return_conditional_losses_1020478�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_dropout_92_layer_call_fn_1020483
,__inference_dropout_92_layer_call_fn_1020488�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_92_layer_call_and_return_conditional_losses_1020493
G__inference_dropout_92_layer_call_and_return_conditional_losses_1020505�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_fc2_layer_call_fn_1020514�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_fc2_layer_call_and_return_conditional_losses_1020525�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1020014input_31"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1020412�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
E__inference_MaxPool2_layer_call_and_return_conditional_losses_1020420b4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0���������� 
� �
*__inference_MaxPool2_layer_call_fn_1020399wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
*__inference_MaxPool2_layer_call_fn_1020404U4�1
*�'
%�"
inputs���������� 
� "����������� �
D__inference_Predict_layer_call_and_return_conditional_losses_1019950s
01:;>�;
4�1
'�$
input_31����������
p 

 
� "%�"
�
0���������
� �
D__inference_Predict_layer_call_and_return_conditional_losses_1019983s
01:;>�;
4�1
'�$
input_31����������
p

 
� "%�"
�
0���������
� �
D__inference_Predict_layer_call_and_return_conditional_losses_1020118q
01:;<�9
2�/
%�"
inputs����������
p 

 
� "%�"
�
0���������
� �
D__inference_Predict_layer_call_and_return_conditional_losses_1020200q
01:;<�9
2�/
%�"
inputs����������
p

 
� "%�"
�
0���������
� �
)__inference_Predict_layer_call_fn_1019656f
01:;>�;
4�1
'�$
input_31����������
p 

 
� "�����������
)__inference_Predict_layer_call_fn_1019917f
01:;>�;
4�1
'�$
input_31����������
p

 
� "�����������
)__inference_Predict_layer_call_fn_1020039d
01:;<�9
2�/
%�"
inputs����������
p 

 
� "�����������
)__inference_Predict_layer_call_fn_1020064d
01:;<�9
2�/
%�"
inputs����������
p

 
� "�����������
"__inference__wrapped_model_1019313o
01:;6�3
,�)
'�$
input_31����������
� ")�&
$
fc2�
fc2����������
J__inference_activation_61_layer_call_and_return_conditional_losses_1020394b4�1
*�'
%�"
inputs���������� 
� "*�'
 �
0���������� 
� �
/__inference_activation_61_layer_call_fn_1020389U4�1
*�'
%�"
inputs���������� 
� "����������� �
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020296|@�=
6�3
-�*
inputs������������������ 
p 
� "2�/
(�%
0������������������ 
� �
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020330|@�=
6�3
-�*
inputs������������������ 
p
� "2�/
(�%
0������������������ 
� �
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020350l8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
S__inference_batch_normalization_61_layer_call_and_return_conditional_losses_1020384l8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
8__inference_batch_normalization_61_layer_call_fn_1020237o@�=
6�3
-�*
inputs������������������ 
p 
� "%�"������������������ �
8__inference_batch_normalization_61_layer_call_fn_1020250o@�=
6�3
-�*
inputs������������������ 
p
� "%�"������������������ �
8__inference_batch_normalization_61_layer_call_fn_1020263_8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
8__inference_batch_normalization_61_layer_call_fn_1020276_8�5
.�+
%�"
inputs���������� 
p
� "����������� �
G__inference_dropout_91_layer_call_and_return_conditional_losses_1020435f8�5
.�+
%�"
inputs���������� 
p 
� "*�'
 �
0���������� 
� �
G__inference_dropout_91_layer_call_and_return_conditional_losses_1020447f8�5
.�+
%�"
inputs���������� 
p
� "*�'
 �
0���������� 
� �
,__inference_dropout_91_layer_call_fn_1020425Y8�5
.�+
%�"
inputs���������� 
p 
� "����������� �
,__inference_dropout_91_layer_call_fn_1020430Y8�5
.�+
%�"
inputs���������� 
p
� "����������� �
G__inference_dropout_92_layer_call_and_return_conditional_losses_1020493\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
G__inference_dropout_92_layer_call_and_return_conditional_losses_1020505\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� 
,__inference_dropout_92_layer_call_fn_1020483O3�0
)�&
 �
inputs���������@
p 
� "����������@
,__inference_dropout_92_layer_call_fn_1020488O3�0
)�&
 �
inputs���������@
p
� "����������@�
@__inference_fc1_layer_call_and_return_conditional_losses_1020478]010�-
&�#
!�
inputs����������(
� "%�"
�
0���������@
� y
%__inference_fc1_layer_call_fn_1020467P010�-
&�#
!�
inputs����������(
� "����������@�
@__inference_fc2_layer_call_and_return_conditional_losses_1020525\:;/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� x
%__inference_fc2_layer_call_fn_1020514O:;/�,
%�"
 �
inputs���������@
� "�����������
G__inference_flatten_30_layer_call_and_return_conditional_losses_1020458^4�1
*�'
%�"
inputs���������� 
� "&�#
�
0����������(
� �
,__inference_flatten_30_layer_call_fn_1020452Q4�1
*�'
%�"
inputs���������� 
� "�����������(�
H__inference_layer_conv2_layer_call_and_return_conditional_losses_1020224f4�1
*�'
%�"
inputs����������
� "*�'
 �
0���������� 
� �
-__inference_layer_conv2_layer_call_fn_1020209Y4�1
*�'
%�"
inputs����������
� "����������� �
%__inference_signature_wrapper_1020014{
01:;B�?
� 
8�5
3
input_31'�$
input_31����������")�&
$
fc2�
fc2���������