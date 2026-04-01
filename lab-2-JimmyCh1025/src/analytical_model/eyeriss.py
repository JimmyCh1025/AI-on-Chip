from dataclasses import dataclass, asdict
from math import ceil

from layer_info import Conv2DShapeParam, MaxPool2DShapeParam

# Memory
DATA_SIZE = 1  # Byte
PSUM_DATA_SIZE = 4  # Byte
BUS_BANDWIDTH = 4  # Byte

# Time
CLOCK_RATE = 200 * 1e6  # 200 MHz
TIME_UNIT = 1  # cycle
SPAD_ACCESS_TIME = 1 * TIME_UNIT
GLB_ACCESS_TIME = 2 * TIME_UNIT
DRAM_ACCESS_TIME = 5 * TIME_UNIT

# Energy
ENERGY_UNIT = 1e-6  # 1 pJ = 10^6 uJ
ENERGY_PER_MAC = 2 * ENERGY_UNIT
ENERGY_PER_GLB_ACCESS = 10 * ENERGY_UNIT
ENERGY_PER_DRAM_ACCESS = 200 * ENERGY_UNIT
POWER_UNIT = 1  # 1 uW
POWER_LEAKAGE = 50 * POWER_UNIT

######################################################################################################
# N: number of ifmaps/ofmaps
# M: number of filters
# H/W: ifmap height/width
# R/S: filter height/width
# E/F: ofmap height/width
# U: stride
#  ----------------------------------------------------------------------------------------------
# m: ofmap channels in global buffer
# n: number of ifmaps in a pass
# e: width of PE-set
# p: number of filters in a pass
# q: (ifmap or filter) channels in a pass
# r: number of PE sets for different (ifmap/filter) channels
# t: number of PE sets for different filters
#  ----------------------------------------------------------------------------------------------
#  Naming Convention
# *_per_pass: compute / storage size required per pass
# *_per_layer: compute / storage size required per layer
######################################################################################################


@dataclass
class EyerissHardwareParam:
    pe_array_h: int
    pe_array_w: int
    ifmap_spad_size: int
    filter_spad_size: int
    psum_spad_size: int
    glb_size: int
    bus_bw: int
    noc_bw: int


@dataclass
class EyerissMappingParam:
    m: int  # number of ofmap channels stored in global buffer
    n: int  # number of ofmaps/ifmaps used in a processing pass
    e: int  # width of the PE set (strip-mined if nessary)
    p: int  # number of filters processed by a PE set
    q: int  # number of ifmap/filter channels processed by a PE set
    r: int  # number of PE sets for different ifmap/filter channels
    t: int  # number of PE sets for different filters


AnalysisResult = dict[str, str | int | float]


class EyerissAnalyzer:
    cnt = 0

    def __init__(
        self,
        name: str | None = None,
        hardware_param: EyerissHardwareParam | None = None,
    ) -> None:
        self.name = name if name is not None else f"mapping_{EyerissAnalyzer.cnt}"
        self._hardware = hardware_param
        self._conv_shape = None
        self._maxpool_shape = None
        self._mapping = None
        EyerissAnalyzer.cnt += 1

    @property
    def hardware(self) -> EyerissHardwareParam:
        return self._hardware

    @hardware.setter
    def hardware(self, hardware_param: EyerissHardwareParam) -> None:
        assert isinstance(hardware_param, EyerissHardwareParam)
        self._hardware = hardware_param

    @property
    def conv_shape(self) -> Conv2DShapeParam:
        return self._conv_shape

    @conv_shape.setter
    def conv_shape(self, conv_param: Conv2DShapeParam) -> None:
        assert isinstance(conv_param, Conv2DShapeParam)
        self._conv_shape = conv_param

    @property
    def maxpool_shape(self) -> MaxPool2DShapeParam:
        return self._maxpool_shape

    @maxpool_shape.setter
    def maxpool_shape(self, maxpool_param: MaxPool2DShapeParam | None) -> None:
        assert isinstance(maxpool_param, (MaxPool2DShapeParam, type(None)))
        self._maxpool_shape = maxpool_param

    @property
    def mapping(self) -> EyerissMappingParam:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping_param: EyerissMappingParam) -> None:
        self._mapping = mapping_param

    # Scratchpad Memory Usage
    def filter_used(self) -> int:
        return self.mapping.q * self.conv_shape.S * self.mapping.p

    def ifmap_used(self) -> int:
        return self.mapping.q * self.conv_shape.S

    def psum_used(self) -> int:
        return self.mapping.p

    @property
    def spad_size_legal(self) -> dict[str, bool]:
        return {
            "ifmap": self.ifmap_used() <= self.hardware.ifmap_spad_size,
            "filter": self.filter_used() <= self.hardware.filter_spad_size,
            "psum": self.psum_used() <= self.hardware.psum_spad_size,
        }

    @property
    def spad_usage(self) -> dict[str, int]:
        return {
            "ifmap": self.ifmap_used(),
            "filter": self.filter_used(),
            "psum": self.psum_used(),
        }

    # Global Buffer (GLB) Usage
    @property
    def glb_usage_per_pass(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        # ifmap: n(ifmap) * qr(ifmap/filter channel per pe * pe set) * [stride *(PE width-1)+filter height] * ifmap width * data size
        sizes["ifmap"] = self.mapping.n * (self.mapping.q * self.mapping.r) * (self.conv_shape.U*(self.mapping.e-1)+self.conv_shape.R) * self.conv_shape.W * DATA_SIZE
        # filter: pass filter totoal (p*t) * channel total (q*r) * kernel size * data size
        sizes["filter"] = (self.mapping.p * self.mapping.t) * (self.mapping.q * self.mapping.r) * self.conv_shape.R * self.conv_shape.S * DATA_SIZE
        # bias: pass filter totoal (p*t) * psum data size
        sizes["bias"] = (self.mapping.p * self.mapping.t) * PSUM_DATA_SIZE
        # psum: n(ifmap) * m(ofmap channel stored in glb) * pe set width * output width * psum data size
        sizes["psum"] = self.mapping.n * self.mapping.m * self.mapping.e * self.conv_shape.F * PSUM_DATA_SIZE
        
        sizes["total"] = sizes["ifmap"] + sizes["filter"] + sizes["bias"] + sizes["psum"]  
        return sizes

    
    @property
    def glb_size_legal(self) -> bool:
        return self.glb_usage_per_pass["total"] <= self.hardware.glb_size

    # DRAM Accesses (DRAM-GLB data movement)
    @property
    def dram_access_per_layer(self) -> dict[str, int]:
        res: dict[str, int] = {}
        #! <<<========= Implement here =========>>>
        
        M_m = ceil(self.conv_shape.M / self.mapping.m)
        E_e = ceil(self.conv_shape.E / self.mapping.e)
        N_n = ceil(self.conv_shape.N / self.mapping.n) 
        C_qr = ceil(self.conv_shape.C / (self.mapping.q * self.mapping.r))
        m_pt = ceil(self.mapping.m / (self.mapping.p * self.mapping.t))

        ifmap_tile_size = self.mapping.n * (self.mapping.q * self.mapping.r) * (self.conv_shape.U*(self.mapping.e-1)+self.conv_shape.R) * self.conv_shape.W * DATA_SIZE   
        filter_tile_size = (self.mapping.p * self.mapping.t) * (self.mapping.q * self.mapping.r) * self.conv_shape.R * self.conv_shape.S * DATA_SIZE
        bias_tile_size = (self.mapping.p * self.mapping.t) * PSUM_DATA_SIZE

        # ifmap : refetch(M_m) * num of tiles(E_e * N_n * C_qr) * tile size(ifmap_tile_size) 
        res["ifmap_read"] = M_m * (E_e * N_n * C_qr) * ifmap_tile_size
        # filter : refetch(E_e * N_n) * num of tiles(M_m * m_pt * C_qr) * tile size(filter_tile_size)
        res["filter_read"] = (E_e * N_n) * (M_m * m_pt * C_qr) * filter_tile_size
        res["bias_read"] =  (E_e * N_n) * (M_m * m_pt) * bias_tile_size
        
        res["read"] = res["ifmap_read"] + res["filter_read"] + res["bias_read"]
        
        # if MaxPool, out_E and out_F is maxpool E and F, else out_E and out_F is conv E and F
        if self.maxpool_shape:
            out_E = self.conv_shape.E // self.maxpool_shape.stride
            out_F = self.conv_shape.F // self.maxpool_shape.stride
        else:
            out_E, out_F = self.conv_shape.E, self.conv_shape.F
            
        res["write"] = self.conv_shape.N * self.conv_shape.M * out_E * out_F * DATA_SIZE
        res["total"] = res["read"] + res["write"]
        return res

    # GLB Accesses (GLB-Spad data movement)
    @property
    def glb_access_per_layer(self) -> dict[str, int]:
        res: dict[str, int] = {}
        #! <<<========= Implement here =========>>>
        M_m = ceil(self.conv_shape.M / self.mapping.m)
        E_e = ceil(self.conv_shape.E / self.mapping.e)
        N_n = ceil(self.conv_shape.N / self.mapping.n) 
        C_qr = ceil(self.conv_shape.C / (self.mapping.q * self.mapping.r))
        m_pt = ceil(self.mapping.m / (self.mapping.p * self.mapping.t))

        ifmap_tile_size = self.mapping.n * (self.mapping.q * self.mapping.r) * (self.conv_shape.U*(self.mapping.e-1)+self.conv_shape.R) * self.conv_shape.W * DATA_SIZE   
        filter_tile_size = (self.mapping.p * self.mapping.t) * (self.mapping.q * self.mapping.r) * self.conv_shape.R * self.conv_shape.S * DATA_SIZE
        bias_tile_size = (self.mapping.p * self.mapping.t) * PSUM_DATA_SIZE
        psum_tile_size = self.mapping.n * (self.mapping.p * self.mapping.t) * self.mapping.e * self.conv_shape.F * PSUM_DATA_SIZE

        res["ifmap_read"] = M_m * (E_e * N_n * C_qr) * m_pt * ifmap_tile_size
        # filter : re
        res["filter_read"] = (E_e * N_n) * (M_m * m_pt * C_qr) * filter_tile_size
        res["bias_read"] =  (E_e * N_n) * (M_m * m_pt) * bias_tile_size

        res["psum_write_conv"] = (E_e * N_n * M_m * m_pt) * C_qr * psum_tile_size
        res["psum_read_conv"] = (E_e * N_n * M_m * m_pt) * (C_qr-1) * psum_tile_size
        
        res["psum_read_pool"] = 0
        res["ofmap_write_pool"] = 0
        
        if self.maxpool_shape:
            out_E = self.conv_shape.E // self.maxpool_shape.stride
            out_F = self.conv_shape.F // self.maxpool_shape.stride
        else:
            out_E, out_F = self.conv_shape.E, self.conv_shape.F
            
        # output activation 
        res["psum_read_pool"] = self.conv_shape.N * self.conv_shape.M * self.conv_shape.E * self.conv_shape.F * PSUM_DATA_SIZE
        res["ofmap_write_pool"] = self.conv_shape.N * self.conv_shape.M * out_E * out_F * DATA_SIZE
        
        res["psum_read"] = res["psum_read_conv"] + res["psum_read_pool"]
        res["psum_write"] = res["psum_write_conv"] + res["ofmap_write_pool"]
        
        res["read"] = res["ifmap_read"] + res["filter_read"] + res["bias_read"] + res["psum_read"]
        res["write"] = res["psum_write"]
        res["total"] = res["read"] + res["write"]
        
        return res

    
    @property
    def latency_per_layer(self) -> int:
        ofmap_size = (
            self.conv_shape.N
            * self.conv_shape.M
            * self.conv_shape.E
            * self.conv_shape.F
        )
        ppu_latency_per_elem = 1 if self.maxpool_shape is None else 5

        return (
            ceil(
                self.glb_access_per_layer["total"]
                * GLB_ACCESS_TIME
                / self.hardware.noc_bw
            )
            + ceil(
                self.dram_access_per_layer["total"]
                * DRAM_ACCESS_TIME
                / self.hardware.bus_bw
            )
            + ofmap_size * ppu_latency_per_elem
        )

    @property
    def macs_per_layer(self) -> int:
        return (
            self.conv_shape.N
            * self.conv_shape.M
            * self.conv_shape.E
            * self.conv_shape.F
            * self.conv_shape.C
            * self.conv_shape.R
            * self.conv_shape.S
        )

    @property
    def energy_per_layer(self) -> dict[str, float]:
        compute_energy = self.macs_per_layer * ENERGY_PER_MAC
        memory_energy = (
            self.glb_access_per_layer["total"] * ENERGY_PER_GLB_ACCESS
            + self.dram_access_per_layer["total"] * ENERGY_PER_DRAM_ACCESS
        )
        leakage_energy = POWER_LEAKAGE * self.latency_per_layer / CLOCK_RATE
        total_energy = compute_energy + memory_energy + leakage_energy
        return {
            "compute": compute_energy,
            "memory": memory_energy,
            "leakage": leakage_energy,
            "total": total_energy,
        }

    @property
    def power_per_layer(self) -> dict[str, float]:
        compute_power = (
            self.energy_per_layer["compute"] / self.latency_per_layer * CLOCK_RATE
        )
        memory_power = (
            self.energy_per_layer["memory"] / self.latency_per_layer * CLOCK_RATE
        )
        leakage_power = POWER_LEAKAGE
        total_power = compute_power + memory_power + leakage_power
        return {
            "compute": compute_power,
            "memory": memory_power,
            "leakage": leakage_power,
            "total": total_power,
        }

    @property
    def operational_intensity(self) -> float:
        return self.macs_per_layer / self.dram_access_per_layer["total"]

    @property
    def peak_performance(self) -> float:
        return self.hardware.pe_array_h * self.hardware.pe_array_w  # MACs per cycle

    @property
    def peak_bandwidth(self) -> float:
        return self.hardware.bus_bw  # bytes per cycle

    @property
    def bound_by(self) -> str:
        machine_blance_point = self.peak_performance / self.peak_bandwidth
        if self.operational_intensity > machine_blance_point:
            return "compute"
        elif self.operational_intensity < machine_blance_point:
            return "memory"
        else:
            return "balanced"

    @property
    def is_compute_bound(self) -> bool:
        return self.bound_by == "compute"

    @property
    def is_memory_bound(self) -> bool:
        return self.bound_by == "memory"

    @property
    def is_balanced(self) -> bool:
        return self.bound_by == "balanced"

    @property
    def summary(self) -> AnalysisResult:
        if not hasattr(self, "mode"):
            self.mode = None

        match self.mode:
            case _:
                return {
                    "layer": self.name,
                    "glb_usage": self.glb_usage_per_pass["total"],  # bytes

                    #"glb_ifmap_read": self.glb_access_per_layer["ifmap_read"],  # bytes
                    #"glb_filter_read": self.glb_access_per_layer["filter_read"],  # bytes
                    #"glb_bias_read": self.glb_access_per_layer["bias_read"],  # bytes
                    #"glb_psum_read_conv": self.glb_access_per_layer["psum_read_conv"],  # bytes
                    #"glb_psum_read_pool": self.glb_access_per_layer["psum_read_pool"],  # bytes
                    #"glb_psum_read": self.glb_access_per_layer["psum_read"],  # bytes
                    #"glb_psum_write_conv": self.glb_access_per_layer["psum_write_conv"],  # bytes
                    #"glb_ofmap_write_pool": self.glb_access_per_layer["ofmap_write_pool"],  # bytes
                    #"glb_psum_write": self.glb_access_per_layer["psum_write"],  # bytes

                    #"glb_read": self.glb_access_per_layer["read"],  # bytes
                    #"glb_write": self.glb_access_per_layer["write"],  # bytes
                    "glb_access": self.glb_access_per_layer["total"],  # bytes

                    #"dram_ifmap_read": self.dram_access_per_layer["ifmap_read"],  # bytes
                    #"dram_filter_read": self.dram_access_per_layer["filter_read"],  # bytes
                    #"dram_bias_read": self.dram_access_per_layer["bias_read"],  # bytes
                    #"dram_read": self.dram_access_per_layer["read"],  # bytes
                    #"dram_write": self.dram_access_per_layer["write"],  # bytes
                    "dram_access": self.dram_access_per_layer["total"],  # bytes
                    #"macs": self.macs_per_layer,
                    "latency": self.latency_per_layer,  # cycles
                    "energy_total": self.energy_per_layer["total"],  # uJ
                    #"power_total": self.power_per_layer["total"],  # uW
                    # or any other metrics you want to include in the report

                    "peak_performance": self.peak_performance, 
                    "peak_bandwidth": self.peak_bandwidth,
                    "intensity": self.operational_intensity
                }
