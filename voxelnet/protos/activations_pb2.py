# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: voxelnet/protos/activations.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='voxelnet/protos/activations.proto',
  package='voxelnet.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1fsecond/protos/activations.proto\x12\rvoxelnet.protos\"\x06\n\x04ReLU\"\x1d\n\tLeakyReLU\x12\x10\n\x08leakness\x18\x01 \x01(\x02\"\x07\n\x05Swish\"\x14\n\x03\x45LU\x12\r\n\x05\x61lpha\x18\x01 \x01(\x02\"+\n\x08Softplus\x12\x0c\n\x04\x62\x65ta\x18\x01 \x01(\x02\x12\x11\n\tthreshold\x18\x02 \x01(\x02\"\n\n\x08Softsign\"\x07\n\x05ReLU6\"\x06\n\x04SELU\"\xdf\x02\n\nActivation\x12#\n\x04relu\x18\x01 \x01(\x0b\x32\x13.voxelnet.protos.ReLUH\x00\x12.\n\nleaky_relu\x18\x02 \x01(\x0b\x32\x18.voxelnet.protos.LeakyReLUH\x00\x12%\n\x05swish\x18\x03 \x01(\x0b\x32\x14.voxelnet.protos.SwishH\x00\x12!\n\x03\x65lu\x18\x04 \x01(\x0b\x32\x12.voxelnet.protos.ELUH\x00\x12+\n\x08softplus\x18\x05 \x01(\x0b\x32\x17.voxelnet.protos.SoftplusH\x00\x12+\n\x08softsign\x18\x06 \x01(\x0b\x32\x17.voxelnet.protos.SoftsignH\x00\x12%\n\x05relu6\x18\x07 \x01(\x0b\x32\x14.voxelnet.protos.ReLU6H\x00\x12#\n\x04selu\x18\x08 \x01(\x0b\x32\x13.voxelnet.protos.SELUH\x00\x42\x0c\n\nactivationb\x06proto3')
)




_RELU = _descriptor.Descriptor(
  name='ReLU',
  full_name='voxelnet.protos.ReLU',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=56,
)


_LEAKYRELU = _descriptor.Descriptor(
  name='LeakyReLU',
  full_name='voxelnet.protos.LeakyReLU',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='leakness', full_name='voxelnet.protos.LeakyReLU.leakness', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=87,
)


_SWISH = _descriptor.Descriptor(
  name='Swish',
  full_name='voxelnet.protos.Swish',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=89,
  serialized_end=96,
)


_ELU = _descriptor.Descriptor(
  name='ELU',
  full_name='voxelnet.protos.ELU',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='alpha', full_name='voxelnet.protos.ELU.alpha', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=98,
  serialized_end=118,
)


_SOFTPLUS = _descriptor.Descriptor(
  name='Softplus',
  full_name='voxelnet.protos.Softplus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='beta', full_name='voxelnet.protos.Softplus.beta', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='threshold', full_name='voxelnet.protos.Softplus.threshold', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=120,
  serialized_end=163,
)


_SOFTSIGN = _descriptor.Descriptor(
  name='Softsign',
  full_name='voxelnet.protos.Softsign',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=165,
  serialized_end=175,
)


_RELU6 = _descriptor.Descriptor(
  name='ReLU6',
  full_name='voxelnet.protos.ReLU6',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=177,
  serialized_end=184,
)


_SELU = _descriptor.Descriptor(
  name='SELU',
  full_name='voxelnet.protos.SELU',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=186,
  serialized_end=192,
)


_ACTIVATION = _descriptor.Descriptor(
  name='Activation',
  full_name='voxelnet.protos.Activation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='relu', full_name='voxelnet.protos.Activation.relu', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='leaky_relu', full_name='voxelnet.protos.Activation.leaky_relu', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='swish', full_name='voxelnet.protos.Activation.swish', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='elu', full_name='voxelnet.protos.Activation.elu', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='softplus', full_name='voxelnet.protos.Activation.softplus', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='softsign', full_name='voxelnet.protos.Activation.softsign', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relu6', full_name='voxelnet.protos.Activation.relu6', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='selu', full_name='voxelnet.protos.Activation.selu', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='activation', full_name='voxelnet.protos.Activation.activation',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=195,
  serialized_end=546,
)

_ACTIVATION.fields_by_name['relu'].message_type = _RELU
_ACTIVATION.fields_by_name['leaky_relu'].message_type = _LEAKYRELU
_ACTIVATION.fields_by_name['swish'].message_type = _SWISH
_ACTIVATION.fields_by_name['elu'].message_type = _ELU
_ACTIVATION.fields_by_name['softplus'].message_type = _SOFTPLUS
_ACTIVATION.fields_by_name['softsign'].message_type = _SOFTSIGN
_ACTIVATION.fields_by_name['relu6'].message_type = _RELU6
_ACTIVATION.fields_by_name['selu'].message_type = _SELU
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['relu'])
_ACTIVATION.fields_by_name['relu'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['leaky_relu'])
_ACTIVATION.fields_by_name['leaky_relu'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['swish'])
_ACTIVATION.fields_by_name['swish'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['elu'])
_ACTIVATION.fields_by_name['elu'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['softplus'])
_ACTIVATION.fields_by_name['softplus'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['softsign'])
_ACTIVATION.fields_by_name['softsign'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['relu6'])
_ACTIVATION.fields_by_name['relu6'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
_ACTIVATION.oneofs_by_name['activation'].fields.append(
  _ACTIVATION.fields_by_name['selu'])
_ACTIVATION.fields_by_name['selu'].containing_oneof = _ACTIVATION.oneofs_by_name['activation']
DESCRIPTOR.message_types_by_name['ReLU'] = _RELU
DESCRIPTOR.message_types_by_name['LeakyReLU'] = _LEAKYRELU
DESCRIPTOR.message_types_by_name['Swish'] = _SWISH
DESCRIPTOR.message_types_by_name['ELU'] = _ELU
DESCRIPTOR.message_types_by_name['Softplus'] = _SOFTPLUS
DESCRIPTOR.message_types_by_name['Softsign'] = _SOFTSIGN
DESCRIPTOR.message_types_by_name['ReLU6'] = _RELU6
DESCRIPTOR.message_types_by_name['SELU'] = _SELU
DESCRIPTOR.message_types_by_name['Activation'] = _ACTIVATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReLU = _reflection.GeneratedProtocolMessageType('ReLU', (_message.Message,), dict(
  DESCRIPTOR = _RELU,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.ReLU)
  ))
_sym_db.RegisterMessage(ReLU)

LeakyReLU = _reflection.GeneratedProtocolMessageType('LeakyReLU', (_message.Message,), dict(
  DESCRIPTOR = _LEAKYRELU,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.LeakyReLU)
  ))
_sym_db.RegisterMessage(LeakyReLU)

Swish = _reflection.GeneratedProtocolMessageType('Swish', (_message.Message,), dict(
  DESCRIPTOR = _SWISH,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.Swish)
  ))
_sym_db.RegisterMessage(Swish)

ELU = _reflection.GeneratedProtocolMessageType('ELU', (_message.Message,), dict(
  DESCRIPTOR = _ELU,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.ELU)
  ))
_sym_db.RegisterMessage(ELU)

Softplus = _reflection.GeneratedProtocolMessageType('Softplus', (_message.Message,), dict(
  DESCRIPTOR = _SOFTPLUS,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.Softplus)
  ))
_sym_db.RegisterMessage(Softplus)

Softsign = _reflection.GeneratedProtocolMessageType('Softsign', (_message.Message,), dict(
  DESCRIPTOR = _SOFTSIGN,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.Softsign)
  ))
_sym_db.RegisterMessage(Softsign)

ReLU6 = _reflection.GeneratedProtocolMessageType('ReLU6', (_message.Message,), dict(
  DESCRIPTOR = _RELU6,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.ReLU6)
  ))
_sym_db.RegisterMessage(ReLU6)

SELU = _reflection.GeneratedProtocolMessageType('SELU', (_message.Message,), dict(
  DESCRIPTOR = _SELU,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.SELU)
  ))
_sym_db.RegisterMessage(SELU)

Activation = _reflection.GeneratedProtocolMessageType('Activation', (_message.Message,), dict(
  DESCRIPTOR = _ACTIVATION,
  __module__ = 'voxelnet.protos.activations_pb2'
  # @@protoc_insertion_point(class_scope:voxelnet.protos.Activation)
  ))
_sym_db.RegisterMessage(Activation)


# @@protoc_insertion_point(module_scope)
