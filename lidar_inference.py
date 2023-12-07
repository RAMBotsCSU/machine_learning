import numpy as np
from pycoral.utils import edgetpu
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":

    model_path = 'lidar_model_quantized_edgetpu.tflite'

    interpreter = edgetpu.make_interpreter(model_path, device='usb')
    interpreter.allocate_tensors()

    print("Allocated tensors")
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Got details")

    # Set input tensor data
    fake_lidar_data = [2106, 2062.5, 2022.5, 2000, 2040.25, 1621.75, 1623.5, 1626, 1620.25, 1613.25, 1608.5, 1604, 1602.25, 1603, 1611.25, 1600.75, 1603.5, 1598.75, 1599.75, 1601.75, 1603.5, 1606.75, 1596, 1605.75, 1610.25, 1615.75, 1619.5, 1626.5, 1633.75, 1639, 1646.75, 1650.5, 1656.5, 1670.25, 1679, 1683.25, 1690.5, 1699.5, 1707.25, 1707.75, 1721.75, 1748.75, 1769.25, 1788.25, 1807.25, 1822.25, 1843.5, 1868.25, 1883.25, 1906.75, 1912, 1814.75, 1833.5, 1746.75, 1701.25, 1665.25, 1648, 1653.25, 1673.5, 1695.25, 1719.5, 1744.25, 1776.75, 1819.75, 1863.5, 1686.5, 1708.5, 1737.75, 2607, 2654.5, 2719.75, 1695.5, 1718, 5563.5, 5514.5, 5469.5, 5412.25, 5364.25, 5301.5, 5236.25, 5203, 5174.5, 5134.75, 5077.5, 5299, 4467.75, 4411.75, 4384.5, 4345.25, 5097.25, 5233, 5204, 4416.5, 4427, 4382.75, 5307.75, 4485, 4418, 4387.5, 4356.5, 4370, 5034, 5170.75, 4418, 4382.25, 4359.75, 4341.75, 4408.5, 4366.25, 4337.75, 4335.5, 4411.5, 4370, 4334.25, 4313, 4390.5, 4354.5, 4328, 4975, 4952, 4935.75, 4378.25, 4995.75, 4967, 4938.5, 4985.75, 5033.75, 4342.25, 5039.25, 4982, 4971.25, 4941.5, 4903.75, 4906.5, 4896.75, 4966, 4924.25, 4954.5, 4963.5, 5020.5, 2147.25, 2445.5, 2071.75, 2043.25, 2006, 1976.5, 1943.25, 1900.75, 1859.5, 1822.25, 1779, 1737.25, 1711, 1690.25, 1670, 1646.75, 1632, 1647, 1677.75, 1711.5, 1751.75, 1628.5, 1871.5, 1930.75, 1979.5, 2023, 2151.75, 1654.25, 1634.5, 2392.75, 1999.75, 2576.5, 1744.75, 2171.5, 1914, 1965.5, 2024.75, 2081.5, 2140, 1651, 2283.5, 1948.25, 4863, 2049.75, 1904.75, 1958, 2010.75, 4802.5, 1662.5, 1934.5, 1988.5, 2374.75, 1683.25, 2162.25, 4909.75, 1984.25, 2035, 2092.5, 2158, 4889.25, 2040, 1766.25, 2163.5, 3350, 1953.75, 1735.5, 2546.25, 4858.5, 1906.25, 1960.75, 2389, 2075, 4876.25, 2783.5, 2086, 2396.75, 2594.75, 2591.25, 2556.5, 2495.25, 2429.25, 2363.5, 2303.75, 2255.25, 2213, 2164, 2119.5, 2081.25, 2039.25, 2235.5, 1828.25, 1795.25, 1769.25, 1732.5, 1703.25, 1692, 1696.25, 1715.25, 1754, 1800.25, 1846, 1884.75, 1911.75, 1946.5, 1779.5, 1818, 1839.75, 2076, 2107.25, 2132.5, 2144.25, 2005.75, 2045.25, 2449.75, 2509.5, 2569.75, 2642.25, 2053.5, 2966.5, 2555.25, 2135.25, 2071.5, 3503, 2917.75, 2481.75, 2544, 3284.5, 2099.5, 2442.5, 2917, 2087, 2684.25, 3275, 2494.25, 2944, 2095.25, 2700.75, 3322, 2901, 2610, 2677.75, 3244, 2902.75, 2589.25, 2652, 2134, 3345, 2927, 2672.25, 11445.5, 11600.5, 10887.5, 11018.25, 11051.75, 11002, 2923.75, 10783.75, 2644.5, 7205.5, 2828.5, 4298.25, 4275.5, 4292.25, 4306.5, 4339, 4352.75, 4274.25, 4088.25, 4132, 4160.25, 4217.5, 4253, 4202.75, 4146.25, 4151.5, 4206.75, 2957.5, 2902, 2835.75, 4147.25, 2679.25, 2623, 2584.75, 2532.75, 2473.75, 2424.75, 2383.5, 2552.75, 2273.5, 2240.25, 2201.25, 2158.75, 2126, 2099.5, 2063, 2028, 2142, 2115.75, 2089.75, 2049.25, 2013, 2170.75, 1884.75, 2108.25, 2065.25, 2024.25, 2018.5, 2125.75, 2087, 2048, 2024.5, 2013.5, 2135.5, 2108, 2065.25, 2026.25, 2002.75, 2215.25, 2016.75, 1998]
    # fake_lidar_data = np.array(fake_lidar_data)
    # fake_lidar_data = fake_lidar_data.reshape(-1, 1)
    # scaler_X = MinMaxScaler()
    # normalized_lidar_view = scaler_X.fit_transform(fake_lidar_data)
    # # normalized_lidar_view = normalized_lidar_view.flatten()
    # normalized_lidar_view = np.expand_dims(normalized_lidar_view, axis=-1)
    

    # input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
    # input_tensor()[0] = normalized_lidar_view


    # Assuming fake_lidar_data is your LiDAR data
    fake_lidar_data = np.array(fake_lidar_data)
    fake_lidar_data = fake_lidar_data.reshape(-1, 1)  # Reshape to a 2D array

    # Initialize MinMaxScaler
    scaler_X = MinMaxScaler()

    # Fit and transform the data using the scaler
    normalized_lidar_view = scaler_X.fit_transform(fake_lidar_data)

    # Reshape to a 1D array
    normalized_lidar_view = normalized_lidar_view.flatten()

    # Assuming 'interpreter' is your TensorFlow Lite interpreter
    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])

    # Set the input tensor with the reshaped normalized LiDAR data
    input_tensor()[0] = normalized_lidar_view

    print("Preprocess complete")

    # Run inference
    interpreter.invoke()

    print("Inference complete")

    # Get the output tensor
    output_tensor = interpreter.tensor(output_details[0]['index'])

    # Get the output values as a NumPy array
    output_values = np.array(output_tensor())

    print("Output:", output_values)



    
