using System.Numerics;
using System.Text.Json;
using RestSharp;
namespace FibonacciAppMVC.Services
{
    public class FibAPIConsumer : IFibAPIConsumerService
    {
        /// <summary>
        /// The following uses an API end point to get the fibonacci number
        /// </summary>
        /// <param name="num">Stringified  BigInteget</param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public async Task<string> consumeAPIAsync(string num)
        {
            try
            {
                var url = "https://nfibonacci20230125232018.azurewebsites.net/?num=" + num;
                // Create a rest client
                var restClient = new RestClient(url);
                var restRequest = new RestRequest();
                restRequest.Method = Method.Get;
                // Execute the request
                var response = await restClient.ExecuteAsync(restRequest);
                // If no response is recieved, then send a message stating that
                if (response == null || response.Content == null)
                    return "API didn't return any response";
                // return the response
                return response.Content;

            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }

        }
    }
}
