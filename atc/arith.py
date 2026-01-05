PREC = 32
FULL = (1 << PREC) - 1
HALF = 1 << (PREC - 1)
Q1 = 1 << (PREC - 2)
Q3 = 3 << (PREC - 2)


class BitWriter:
    def __init__(self):
        self.buf = 0
        self.nbits = 0
        self.bytes = bytearray()

    def write_bit(self, bit: int):
        self.buf = (self.buf << 1) | (bit & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.bytes.append(self.buf & 0xFF)
            self.buf = 0
            self.nbits = 0

    def write_pending(self, bit: int, count: int):
        for _ in range(count):
            self.write_bit(bit ^ 1)

    def flush(self):
        if self.nbits:
            self.buf <<= 8 - self.nbits
            self.bytes.append(self.buf & 0xFF)
            self.buf = 0
            self.nbits = 0
        return bytes(self.bytes)


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.buf = 0
        self.nbits = 0

    def read_bit(self) -> int:
        if self.nbits == 0:
            self.buf = self.data[self.pos] if self.pos < len(self.data) else 0
            self.pos += 1
            self.nbits = 8
        bit = (self.buf >> 7) & 1
        self.buf = (self.buf << 1) & 0xFF
        self.nbits -= 1
        return bit


class Model:
    def __init__(self, n: int, max_total: int = 1 << 15):
        self.n = n
        self.freq = [1] * n
        self.cum = [0] * (n + 1)
        self.total = 0
        self.max_total = max_total
        self._rebuild()

    def _rebuild(self):
        s = 0
        for i in range(self.n):
            self.cum[i] = s
            s += self.freq[i]
        self.cum[self.n] = s
        self.total = s

    def update(self, sym: int):
        self.freq[sym] += 1
        if self.total >= self.max_total:
            for i in range(self.n):
                self.freq[i] = max(1, self.freq[i] >> 1)
        self._rebuild()


class Encoder:
    def __init__(self):
        self.low = 0
        self.high = FULL
        self.pending = 0
        self.bw = BitWriter()

    def encode(self, model: Model, sym: int):
        total = model.total
        lowc = model.cum[sym]
        highc = model.cum[sym + 1]
        rng = self.high - self.low + 1
        self.high = self.low + (rng * highc // total) - 1
        self.low = self.low + (rng * lowc // total)
        while True:
            if self.high < HALF:
                self.bw.write_bit(0)
                self.bw.write_pending(0, self.pending)
                self.pending = 0
            elif self.low >= HALF:
                self.bw.write_bit(1)
                self.bw.write_pending(1, self.pending)
                self.pending = 0
                self.low -= HALF
                self.high -= HALF
            elif self.low >= Q1 and self.high < Q3:
                self.pending += 1
                self.low -= Q1
                self.high -= Q1
            else:
                break
            self.low = (self.low << 1) & FULL
            self.high = ((self.high << 1) | 1) & FULL
        model.update(sym)

    def finish(self):
        self.pending += 1
        if self.low < Q1:
            self.bw.write_bit(0)
            self.bw.write_pending(0, self.pending)
        else:
            self.bw.write_bit(1)
            self.bw.write_pending(1, self.pending)
        return self.bw.flush()


class Decoder:
    def __init__(self, data: bytes):
        self.low = 0
        self.high = FULL
        self.br = BitReader(data)
        self.code = 0
        for _ in range(PREC):
            self.code = ((self.code << 1) | self.br.read_bit()) & FULL

    def decode(self, model: Model) -> int:
        total = model.total
        rng = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // rng
        sym = 0
        while model.cum[sym + 1] <= value:
            sym += 1
        lowc = model.cum[sym]
        highc = model.cum[sym + 1]
        self.high = self.low + (rng * highc // total) - 1
        self.low = self.low + (rng * lowc // total)
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= Q1 and self.high < Q3:
                self.low -= Q1
                self.high -= Q1
                self.code -= Q1
            else:
                break
            self.low = (self.low << 1) & FULL
            self.high = ((self.high << 1) | 1) & FULL
            self.code = ((self.code << 1) | self.br.read_bit()) & FULL
        model.update(sym)
        return sym
