
import torch
import pynvml as pn
from typing import Union


class Device:
    r"""
    Class of tensor default device.
    """
    __DEVICE = torch.device("cpu")
    device_desc = "cpu"

    @classmethod
    def set_device_for_class(cls, __device: Union[torch.device, str, int]):
        r"""
        Set device of tensor based on `device`.

        NOTE: THIS WILL CHANGE `DEVICE` IN ALL INSTANCE OF THIS CLASS.

        Parameters
        ----------
            __device : Union[torch.device, str, int]
                A device instance or a device description like:
                - `"auto"`: Returns the device, GPU first, with the smallest memory usage;
                - number like `"1"`, `1`: Returns the GPU device with the given index if exist, otherwise same as `auto`;
                - `"cpu"`: Return a CPU device.
        """
        if isinstance(__device, torch.device):
            # for device instance
            cls.__DEVICE = __device
        elif isinstance(__device, (str, int)):
            # for device description
            cls.device_desc = cls.__format_desc(__device)
            cls.__DEVICE = torch.device(cls.device_desc)
        else:
            error(f"Set Device For Class [{cls.__name__}] Error", stack_level=2)

    def set_device_for_instance(self, __device: Union[torch.device, str, int]):
        r"""
        Set device of tensor based on `device`.

        Parameters
        ----------
            __device : Union[torch.device, str, int]
                A device instance or a device description like:
                - `"auto"`: Returns the device, GPU first, with the smallest memory usage;
                - number like `"1"`, `1`: Returns the GPU device with the given index if exist, otherwise same as `auto`;
                - `"cpu"`: Return a CPU device.
        """
        if isinstance(__device, torch.device):
            self.__DEVICE = __device
        elif isinstance(__device, (str, int)):
            self.device_desc = self.__format_desc(__device)
            self.__DEVICE = torch.device(self.device_desc)
        else:
            error("Set Device For Instance Error", stack_level=2)

    @classmethod
    def gpu_state(cls, display: bool = True):
        r"""
        Get the state of GPU.

        Parameters
        ----------
            display : bool, default to `True`
                Control whether display state.

        Returns
        -------
            str
                State of GPU.
        """
        def max_size(_list: list[str]):
            return len(max(_list, key=lambda i: len(i)))

        if cls.__DEVICE.type == "cuda":
            text = ""

            try:
                pn.nvmlInit()
                gpu_id_list = ["GPU-ID"]
                gpu_name_list = ["GPU-Name"]
                gpu_menory_list = ["Memory"]
                gpu_utili_list = ["Utilization"]
                gpu_temperature_list = ["Temperature"]

                for i in range(pn.nvmlDeviceGetCount()):
                    handle = pn.nvmlDeviceGetHandleByIndex(i)
                    gpu_info = pn.nvmlDeviceGetMemoryInfo(handle)
                    gpu_free = gpu_info.free/2**30  # GB
                    gpu_total = gpu_info.total/2**30  # GB
                    utilization = (1-gpu_free/gpu_total)*100
                    gpu_id_list.append(str(i))
                    gpu_name_list.append(str(pn.nvmlDeviceGetName(handle).decode('utf-8')))
                    gpu_menory_list.append(f"{'%.1f' % (gpu_total-gpu_free)}GB/{'%.1f' % gpu_total}GB")
                    gpu_utili_list.append(f"{'%.1f' % utilization}%")
                    gpu_temperature_list.append(f"{pn.nvmlDeviceGetTemperature(handle,0)}°C")

                # max_size = lambda _list: len(max(_list, key=lambda i: len(i)))
                for gpu_id, gpu_name, gpu_menory, gpu_utili, gpu_temperature in zip(gpu_id_list, gpu_name_list, gpu_menory_list, gpu_utili_list, gpu_temperature_list):
                    is_selected = gpu_id == str(cls.__DEVICE.index)
                    # gpu_id
                    gpu_id = gpu_id.center(max_size(gpu_id_list))
                    # gpu_id = color_str(gpu_id, fore="d", back="w", styles={"bold"}) if is_selected else gpu_id
                    # gpu_name
                    gpu_name = gpu_name.center(max_size(gpu_name_list))
                    # gpu_menory & gpu_utili
                    gpu_menory = gpu_menory.center(max_size(gpu_menory_list))

                    if is_selected:
                        utili_text = gpu_utili.split(".")[0]
                        usable = utili_text.isdigit() and int(utili_text) < 95
                        gpu_utili = gpu_utili.center(max_size(gpu_utili_list))
                        # gpu_menory = color_str(gpu_menory, fore="d", back="g" if usable else "r", styles={"bold"})
                        # gpu_utili = color_str(gpu_utili, fore="d", back="g" if usable else "r", styles={"bold"})
                        if not usable:
                            print("May Not Enough GPU RAM For Training, Lower Batch Size Suggested")
                    else:
                        gpu_utili = gpu_utili.center(max_size(gpu_utili_list))
                    # gpu_temperature
                    gpu_temperature = gpu_temperature.center(max_size(gpu_temperature_list))

                    text += f"     | {gpu_id} | {gpu_name} | {gpu_menory} | {gpu_utili} | {gpu_temperature} |\n"

            except Exception as e:
                print("GPU State Not Available", e)
                text = ""

        else:
            text = ""

        if display:
            print(f"Device Activate [{cls.__DEVICE}]")
            if text:
                print(text)

        return text

    @staticmethod
    def __format_desc(__device_desc: Union[str, int] = "auto"):
        r"""
        Format the description of device based on `device_desc`.

        Parameters
        ----------
            __device_desc : str, default to `"auto"`
                The description of device.
                - `"auto"`: Returns the device, GPU first, with the smallest memory usage;
                - number like `"1"`, `1`: Returns the GPU device with the given index if exist, otherwise same as `auto`;
                - `"cpu"`: Return a CPU device.

        Returns
        -------
            str
                The formatted description of device.
        """
        __device_desc = str(__device_desc).lower()

        if __device_desc == "cpu":
            formatted_desc = "cpu"
        else: 
            # Try GPU
            if torch.cuda.is_available():
                # CUDA is available
                try:
                    pn.nvmlInit()
                    cuda_range = range(pn.nvmlDeviceGetCount())
                    if __device_desc.isdigit() and int(__device_desc) in cuda_range:
                        # for the given device index
                        formatted_desc = f"cuda:{__device_desc}"

                    else:
                        # for auto
                        if __device_desc.isdigit():
                            print(f"GPU Index [{__device_desc}] Out of Range [{list(cuda_range)}], Set As [auto]")
                            __device_desc = "auto"
                        if __device_desc != "auto":
                            print(f"Invalid Device [{__device_desc}], Set As [auto]")

                        free_info_list = []
                        for i in cuda_range:
                            gpu_info = pn.nvmlDeviceGetMemoryInfo(pn.nvmlDeviceGetHandleByIndex(i))
                            free_info_list.append(gpu_info.free)
                        cuda_index = free_info_list.index(max(free_info_list))
                        formatted_desc = f"cuda:{cuda_index}"

                except Exception as e:
                    print("[pynvml] Is Not Available, Try To Set Device Unsafely", e)

                    if __device_desc.isdigit():
                        formatted_desc = f"cuda:{__device_desc}"
                    else:
                        formatted_desc = "cuda"

            else:
                # CUDA is not available
                print("CUDA Is Not Available, Set Device As [cpu]")
                formatted_desc = "cpu"

        return formatted_desc

    @property
    def DEVICE(self):
        r"""
        Device of tensor.
        """
        return self.__DEVICE
