using FibonacciAppMVC.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using FibonacciAppMVC.Services;
using System.Text.Json;
using RestSharp;
using System;
using System.Numerics;

namespace FibonacciAppMVC.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly ISearchHistoryService _historyService;
        private readonly IFibAPIConsumerService _apiConsumer;
        public HomeController(ILogger<HomeController> logger, ISearchHistoryService historyService, IFibAPIConsumerService apiConsumer)
        {
            _logger = logger;
            _historyService = historyService;
            _apiConsumer = apiConsumer;
        }
        /// <summary>
        /// The index method for the controller
        /// </summary>
        /// <returns>view</returns>
        public IActionResult Index()
        {
            return View();
        }
        /// <summary>
        /// This method returns a serialized records of all the fibonacci searches
        /// </summary>
        /// <returns></returns>
        public async Task<string> GetHistory()
        {
            // Get the search history using the SearchHistory service
            var result = await _historyService.GetSearchHistoryAsync();
            return JsonSerializer.Serialize(result);
        }

        /// <summary>
        /// The function uses the _apiConsumer service to get the fibonacci number
        /// </summary>
        /// <param name="num">This param reperesents the position in the fibonacci sequence</param>
        /// <returns>The fibonacci number converted to string</returns>
        public async Task<string> GetNthFib(string num)
        {
            BigInteger result;
            // Parse the string to BigInteger. If parse fails then return the invalid input message
            if (!BigInteger.TryParse(num, out result))
                return "Invalid integer passed";
            // If the position is less than 1,then it's invalid as the first position in fibonacci sequence is 1 which is number 0. There is no 0th position. 
            if (result < 1)
                return "Passed integer should be greater than 0";
            // Save the searched number in the history
            await _historyService.SaveSearchHistoryAsync(result);
            // use the _apiConsumer service to call an external API which returns the fibonacci number at a specific position
            return await _apiConsumer.consumeAPIAsync(result.ToString());
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}