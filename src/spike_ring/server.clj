(ns spike-ring.server
  (:require [immutant.web :as web]
            [ring.middleware.defaults :refer :all]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [ring.middleware.edn :refer :all]
            [mount.core :refer [defstate]]))

(defn app
  [request]
  (println (:params request))
  {:status 200
   :body   (pr-str [{:hello "world!"} {:hi "everyone"}])})

;(web/run (wrap-defaults app api-defaults))
(def start
  (partial web/run
           (-> app
               wrap-edn-params)
           {:host "0.0.0.0"}))

(def stop
  (partial web/stop))

(defstate server
          :start (start)
          :stop (stop))
