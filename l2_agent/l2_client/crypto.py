"""
Lineage2 C6 Interlude -- cryptographic primitives and low-level network I/O.

The MIT License (MIT)

Copyright (c) 2026 Viacheslav Reutskyi <viacheslav.reutskyi@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import socket
import struct

from Crypto.Cipher import Blowfish

L2MOBIUS_STATIC_KEY = b'\x6b\x60\xcb\x5b\x82\xce\x90\xb1\xcc\x2b\x6c\x55\x6c\x6c\x6c\x6c'
"""Static 16-byte blowfish key used for the initial login-server packet exchange."""


def blowfish_ecb_crypt(data: bytearray, key: bytes, encrypt: bool) -> bytearray:
    """
    Blowfish ECB encrypt/decrypt with LE ↔ BE conversion to match L2J ``BlowfishEngine``.

    L2J stores 64-bit blocks in *big-endian* order before feeding them to the
    Blowfish cipher, and swaps back to little-endian afterwards.  This function
    replicates that behaviour in-place.

    :param data: buffer whose length is a multiple of 8 bytes
    :param key:  blowfish key (4-56 bytes)
    :param encrypt: ``True`` to encrypt, ``False`` to decrypt
    :returns: *data* (modified in-place) for convenience
    """
    be = bytearray(len(data))
    for i in range(0, len(data), 8):
        w0, w1 = struct.unpack_from("<II", data, i)
        struct.pack_into(">II", be, i, w0, w1)
    c = Blowfish.new(key, Blowfish.MODE_ECB)
    r = c.encrypt(bytes(be)) if encrypt else c.decrypt(bytes(be))
    for i in range(0, len(data), 8):
        w0, w1 = struct.unpack_from(">II", r, i)
        struct.pack_into("<II", data, i, w0, w1)
    return data


def unscramble_modulus(mod: bytearray):
    """
    Unscramble the 128-byte RSA modulus received in the init packet (opcode ``0x00``).

    The original L2 client scrambles the modulus before sending it; this mirrors
    the server-side ``unscrambleModulus`` method in
    ``org.l2jmobius.commons.crypt.NewCrypt``.

    :param mod: 128-byte bytearray -- modified in-place
    """
    for i in range(0x40):
        mod[0x40 + i] ^= mod[i]
    for i in range(4):
        mod[0x0d + i] ^= mod[0x34 + i]
    for i in range(0x40):
        mod[i] ^= mod[0x40 + i]
    for i in range(4):
        mod[0x00 + i], mod[0x4d + i] = mod[0x4d + i], mod[0x00 + i]


def dec_xor_pass(b: bytearray):
    """
    Second-stage decryption used by the login-server init packet.

    Starting from the end of the payload, every 4-byte word is XOR-decrypted
    with a rolling key derived from the previous plain-text word.

    :param b: decrypted Blowfish output -- modified in-place
    :returns: *b* for convenience
    """
    s, sp, pk = len(b), len(b) - 8, struct.unpack_from("<I", b, len(b) - 8)[0]
    cp = sp - 4
    while cp >= 4:
        ev = struct.unpack_from("<I", b, cp)[0]
        dv = ev ^ pk
        pk = (pk - dv) & 0xFFFFFFFF
        struct.pack_into("<I", b, cp, dv)
        cp -= 4
    return b


def append_l2_checksum(b: bytearray):
    """
    Calculate and append the L2 packet checksum (XOR of all preceding DWORDs).

    Writes the 4-byte checksum into the last 4 bytes of the buffer.  The caller
    must have already reserved space (e.g. by padding with four zero-bytes).

    :param b: bytearray whose last 4 bytes will be overwritten
    """
    s, cs, cnt = len(b), 0, len(b) - 4
    for i in range(0, cnt, 4):
        cs ^= struct.unpack_from("<I", b, i)[0]
    struct.pack_into("<I", b, cnt, cs)


def decrypt_init_packet(raw: bytes, key: bytes) -> bytearray:
    """
    Full two-stage decryption of the login-server init packet (opcode ``0x00``).

    1. Blowfish-ECB decrypt with the static key.
    2. Rolling-XOR decrypt the result.

    :param raw: raw bytes received from the login server
    :param key: 16-byte blowfish key (``L2MOBIUS_STATIC_KEY``)
    :returns: decrypted bytearray
    """
    b = bytearray(raw)
    blowfish_ecb_crypt(b, key, False)
    return dec_xor_pass(b)


def decrypt_l2_packet(raw: bytes, key: bytes) -> bytearray:
    """
    Blowfish-ECB decrypt a standard (post-login) L2 packet.

    :param raw: raw bytes received from the server
    :param key: session blowfish key (received in the init packet)
    :returns: decrypted bytearray
    """
    b = bytearray(raw)
    return blowfish_ecb_crypt(b, key, False)


def encrypt_l2_packet(raw: bytes, key: bytes) -> bytes:
    """
    Encrypt and frame a standard L2 packet for sending.

    Steps:
    1. Pad to 8-byte alignment + 4 extra zero-bytes for the checksum.
    2. Compute and append the XOR checksum.
    3. Blowfish-ECB encrypt.
    4. Prepend a 2-byte little-endian length header.

    :param raw: plaintext packet body (without the 2-byte length header)
    :param key: session blowfish key
    :returns: fully-framed, encrypted bytes ready for ``sendall()``
    """
    b = bytearray(raw)
    b.extend(b'\x00' * 4)
    if len(b) % 8:
        b.extend(b'\x00' * (8 - len(b) % 8))
    append_l2_checksum(b)
    blowfish_ecb_crypt(b, key, True)
    return struct.pack("<H", len(b) + 2) + b


def send_encrypted(sock: socket.socket, body: bytes, key: bytes):
    """
    Convenience wrapper: encrypt *body* with *key* and send over *sock*.

    :param sock: connected TCP socket
    :param body: plaintext packet body
    :param key: session blowfish key
    """
    sock.sendall(encrypt_l2_packet(body, key))


def recv_packet(sock: socket.socket) -> bytes:
    """
    Read a single L2 packet from the socket.

    Reads the 2-byte length header, then loops until the entire payload has
    been received.  Does **not** decrypt -- returns raw bytes.

    :param sock: connected TCP socket
    :raises ConnectionError: if the header or payload cannot be read completely
    :returns: raw packet body (without the 2-byte length header)
    """
    h = sock.recv(2)
    if len(h) == 1:  # TCP may split the header too
        h += sock.recv(1)
    if len(h) != 2:
        raise ConnectionError("recv header")
    sz = struct.unpack_from("<H", h)[0]
    p = bytearray()
    while len(p) < sz - 2:
        c = sock.recv(sz - 2 - len(p))
        if not c:
            raise ConnectionError("recv payload")
        p.extend(c)
    return bytes(p)


def send_raw(sock: socket.socket, data: bytes):
    """
    Send raw bytes over the socket (no encryption, no length header).

    Used for game-server packets that use the plain protocol (opcodes
    ``0x00`` ProtocolVersion, ``0x0D`` CharacterSelect, ``0x03`` EnterWorld,
    etc.) where the two-byte length header is already included by
    ``packets.pack()``.

    :param sock: connected TCP socket
    :param data: bytes to send
    """
    sock.sendall(data)
