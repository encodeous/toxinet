using System;
using toxinet;

namespace ToxiNetTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var net = ToxiNet.GetToxiNet();
            while (true)
            {
                var input = Console.ReadLine();

                foreach (var k in net.Predict(input))
                {
                    Console.WriteLine($"Type: {k.PredictionType.ToString()} Percent: {k.Prediction}");
                }
            }
        }
    }
}
