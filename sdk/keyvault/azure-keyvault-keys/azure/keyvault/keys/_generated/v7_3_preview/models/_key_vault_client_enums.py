# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from enum import Enum, EnumMeta
from six import with_metaclass

class _CaseInsensitiveEnumMeta(EnumMeta):
    def __getitem__(self, name):
        return super().__getitem__(name.upper())

    def __getattr__(cls, name):
        """Return the enum member matching `name`
        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.
        """
        try:
            return cls._member_map_[name.upper()]
        except KeyError:
            raise AttributeError(name)


class ActionType(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """The type of the action.
    """

    #: Rotate the key based on the key policy.
    ROTATE = "rotate"
    #: Trigger event grid events. For preview, the notification time is not configurable and it is
    #: default to 30 days before expiry.
    NOTIFY = "notify"

class DeletionRecoveryLevel(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """Reflects the deletion recovery level currently in effect for keys in the current vault. If it
    contains 'Purgeable' the key can be permanently deleted by a privileged user; otherwise, only
    the system can purge the key, at the end of the retention interval.
    """

    #: Denotes a vault state in which deletion is an irreversible operation, without the possibility
    #: for recovery. This level corresponds to no protection being available against a Delete
    #: operation; the data is irretrievably lost upon accepting a Delete operation at the entity level
    #: or higher (vault, resource group, subscription etc.).
    PURGEABLE = "Purgeable"
    #: Denotes a vault state in which deletion is recoverable, and which also permits immediate and
    #: permanent deletion (i.e. purge). This level guarantees the recoverability of the deleted entity
    #: during the retention interval (90 days), unless a Purge operation is requested, or the
    #: subscription is cancelled. System wil permanently delete it after 90 days, if not recovered.
    RECOVERABLE_PURGEABLE = "Recoverable+Purgeable"
    #: Denotes a vault state in which deletion is recoverable without the possibility for immediate
    #: and permanent deletion (i.e. purge). This level guarantees the recoverability of the deleted
    #: entity during the retention interval(90 days) and while the subscription is still available.
    #: System wil permanently delete it after 90 days, if not recovered.
    RECOVERABLE = "Recoverable"
    #: Denotes a vault and subscription state in which deletion is recoverable within retention
    #: interval (90 days), immediate and permanent deletion (i.e. purge) is not permitted, and in
    #: which the subscription itself  cannot be permanently canceled. System wil permanently delete it
    #: after 90 days, if not recovered.
    RECOVERABLE_PROTECTED_SUBSCRIPTION = "Recoverable+ProtectedSubscription"
    #: Denotes a vault state in which deletion is recoverable, and which also permits immediate and
    #: permanent deletion (i.e. purge when 7<= SoftDeleteRetentionInDays < 90). This level guarantees
    #: the recoverability of the deleted entity during the retention interval, unless a Purge
    #: operation is requested, or the subscription is cancelled.
    CUSTOMIZED_RECOVERABLE_PURGEABLE = "CustomizedRecoverable+Purgeable"
    #: Denotes a vault state in which deletion is recoverable without the possibility for immediate
    #: and permanent deletion (i.e. purge when 7<= SoftDeleteRetentionInDays < 90).This level
    #: guarantees the recoverability of the deleted entity during the retention interval and while the
    #: subscription is still available.
    CUSTOMIZED_RECOVERABLE = "CustomizedRecoverable"
    #: Denotes a vault and subscription state in which deletion is recoverable, immediate and
    #: permanent deletion (i.e. purge) is not permitted, and in which the subscription itself cannot
    #: be permanently canceled when 7<= SoftDeleteRetentionInDays < 90. This level guarantees the
    #: recoverability of the deleted entity during the retention interval, and also reflects the fact
    #: that the subscription itself cannot be cancelled.
    CUSTOMIZED_RECOVERABLE_PROTECTED_SUBSCRIPTION = "CustomizedRecoverable+ProtectedSubscription"

class JsonWebKeyCurveName(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """Elliptic curve name. For valid values, see JsonWebKeyCurveName.
    """

    #: The NIST P-256 elliptic curve, AKA SECG curve SECP256R1.
    P256 = "P-256"
    #: The NIST P-384 elliptic curve, AKA SECG curve SECP384R1.
    P384 = "P-384"
    #: The NIST P-521 elliptic curve, AKA SECG curve SECP521R1.
    P521 = "P-521"
    #: The SECG SECP256K1 elliptic curve.
    P256_K = "P-256K"

class JsonWebKeyEncryptionAlgorithm(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """algorithm identifier
    """

    RSA_OAEP = "RSA-OAEP"
    RSA_OAEP256 = "RSA-OAEP-256"
    RSA1_5 = "RSA1_5"
    A128_GCM = "A128GCM"
    A192_GCM = "A192GCM"
    A256_GCM = "A256GCM"
    A128_KW = "A128KW"
    A192_KW = "A192KW"
    A256_KW = "A256KW"
    A128_CBC = "A128CBC"
    A192_CBC = "A192CBC"
    A256_CBC = "A256CBC"
    A128_CBCPAD = "A128CBCPAD"
    A192_CBCPAD = "A192CBCPAD"
    A256_CBCPAD = "A256CBCPAD"

class JsonWebKeyOperation(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """JSON web key operations. For more information, see JsonWebKeyOperation.
    """

    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    WRAP_KEY = "wrapKey"
    UNWRAP_KEY = "unwrapKey"
    IMPORT_ENUM = "import"
    EXPORT = "export"

class JsonWebKeySignatureAlgorithm(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """The signing/verification algorithm identifier. For more information on possible algorithm
    types, see JsonWebKeySignatureAlgorithm.
    """

    #: RSASSA-PSS using SHA-256 and MGF1 with SHA-256, as described in
    #: https://tools.ietf.org/html/rfc7518.
    PS256 = "PS256"
    #: RSASSA-PSS using SHA-384 and MGF1 with SHA-384, as described in
    #: https://tools.ietf.org/html/rfc7518.
    PS384 = "PS384"
    #: RSASSA-PSS using SHA-512 and MGF1 with SHA-512, as described in
    #: https://tools.ietf.org/html/rfc7518.
    PS512 = "PS512"
    #: RSASSA-PKCS1-v1_5 using SHA-256, as described in https://tools.ietf.org/html/rfc7518.
    RS256 = "RS256"
    #: RSASSA-PKCS1-v1_5 using SHA-384, as described in https://tools.ietf.org/html/rfc7518.
    RS384 = "RS384"
    #: RSASSA-PKCS1-v1_5 using SHA-512, as described in https://tools.ietf.org/html/rfc7518.
    RS512 = "RS512"
    #: Reserved.
    RSNULL = "RSNULL"
    #: ECDSA using P-256 and SHA-256, as described in https://tools.ietf.org/html/rfc7518.
    ES256 = "ES256"
    #: ECDSA using P-384 and SHA-384, as described in https://tools.ietf.org/html/rfc7518.
    ES384 = "ES384"
    #: ECDSA using P-521 and SHA-512, as described in https://tools.ietf.org/html/rfc7518.
    ES512 = "ES512"
    #: ECDSA using P-256K and SHA-256, as described in https://tools.ietf.org/html/rfc7518.
    ES256_K = "ES256K"

class JsonWebKeyType(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """JsonWebKey Key Type (kty), as defined in
    https://tools.ietf.org/html/draft-ietf-jose-json-web-algorithms-40.
    """

    #: Elliptic Curve.
    EC = "EC"
    #: Elliptic Curve with a private key which is stored in the HSM.
    EC_HSM = "EC-HSM"
    #: RSA (https://tools.ietf.org/html/rfc3447).
    RSA = "RSA"
    #: RSA with a private key which is stored in the HSM.
    RSA_HSM = "RSA-HSM"
    #: Octet sequence (used to represent symmetric keys).
    OCT = "oct"
    #: Octet sequence (used to represent symmetric keys) which is stored the HSM.
    OCT_HSM = "oct-HSM"

class KeyEncryptionAlgorithm(with_metaclass(_CaseInsensitiveEnumMeta, str, Enum)):
    """The encryption algorithm to use to protected the exported key material
    """

    CKM_RSA_AES_KEY_WRAP = "CKM_RSA_AES_KEY_WRAP"
    RSA_AES_KEY_WRAP256 = "RSA_AES_KEY_WRAP_256"
    RSA_AES_KEY_WRAP384 = "RSA_AES_KEY_WRAP_384"
