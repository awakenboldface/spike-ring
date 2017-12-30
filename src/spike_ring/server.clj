(ns spike-ring.server
  (:require [immutant.web :as web]
            [ring.middleware.defaults :refer :all]
            [ring.middleware.json :refer [wrap-json-response wrap-json-body]]
            [ring.middleware.edn :refer :all]
            [ring.middleware.cors :refer [wrap-cors]]
            [clj-http.client :as client]
            [clojure.walk :as walk]
            [aid.core :as aid]
            [com.rpl.specter :as s]
            [cats.monad.maybe :as maybe]
            [clojure.string :as str]
            [clojure.set :as set]
            [mount.core :refer [defstate]]))

(defn parse-number
  [x]
  (if (and (string? x) (re-find #"^\d+$" x))
    (read-string (re-find #"^\d+$" x))
    x))

(defn infer
  [coll]
  (walk/prewalk parse-number
                (:body (client/post "http://localhost:5000"
                                    {:content-type :json
                                     :form-params  {:action "infer"
                                                    :inputs coll}
                                     :as           :json}))))

(defn parse
  [s]
  (walk/prewalk parse-number
                (:body (client/post "http://localhost:5000"
                                    {:content-type :json
                                     :form-params  {:action "parse"
                                                    :input  s}
                                     :as           :json}))))


(def tags
  #{"NNS" "NNPS" "VBZ"})

(defn uninflect-open
  [token]
  (cond (tags (:tag_ token)) (:lemma_ token)
        (= (:lemma_ token) "be") "is"
        :else (:lower_ token)))

(def countable
  {"few"   "little"
   "fewer" "less"
   "many"  "much"})

(def plural
  {"these" "this"
   "those" "that"})

(def were
  {"were" "was"})

(def word
  (merge countable plural were))

(def uninflect-closed
  (comp (partial apply get word)
        (partial repeat 2)))

(def articles
  (sorted-set "a" "an" "the"))

(def remove-dash
  (partial reduce
           (fn [reduction element]
             (if (and (articles (last reduction)) (= element "-"))
               reduction
               (conj reduction element)))
           []))


(def get-inputs
  (comp remove-dash
        (partial map (comp uninflect-closed
                           uninflect-open))))

(def remove-dash-map
  (partial reduce
           (fn [reduction element]
             (if (and (articles (:lemma_ (last reduction))) (= (:lemma_ element) "-"))
               (s/setval [s/LAST :dash-removed] true reduction)
               (s/setval s/END [(s/setval :dash-removed false element)] reduction)))
           []))

(def get-dash-removeds
  ;TODO extract shift
  (comp drop-last
        (partial cons false)
        (partial map :dash-removed)
        remove-dash-map))

(def article?
  (comp (partial contains? articles)
        :lower_))

(def get-article-removeds
  (comp drop-last
        (partial cons false)
        (partial map article?)
        remove-dash-map))

(def get-article-titles
  (comp drop-last
        (partial cons false)
        (partial map (aid/build and
                                article?
                                :is_title))
        remove-dash-map))

(def proper-nouns
  #{"NNP" "NNPS"})

(def get-uppers
  (comp (partial map (aid/build and
                                :is_upper
                                (comp not proper-nouns :tag_)))
        remove-dash-map))

(def get-starts
  (comp (partial s/setval* s/FIRST true)
        (partial map (comp not nil? :is_sent_start))
        remove-dash-map))

(def supplement-parsed
  (comp (partial remove (comp articles :lower_))
        (partial apply
                 map
                 (fn [m
                      dash-removed
                      article-removed
                      article-title
                      upper
                      start]
                   (merge m {:dash-removed    dash-removed
                             :article-removed article-removed
                             :article-title   article-title
                             :upper           upper
                             :start           start})))
        (juxt remove-dash-map
              get-dash-removeds
              get-article-removeds
              get-article-titles
              get-uppers
              get-starts)))

(def article
  {0 (maybe/nothing)
   1 (maybe/just "a")
   2 (maybe/just "an")
   3 (maybe/just "the")})

(defn merge-inferred
  [inferred coll]
  (map (fn [m* article* inflected]
         (merge m*
                {:article (article article*)}
                {:inflected (not (zero? inflected))}))
       coll
       (:articles inferred)
       (:inflecteds inferred)))

(defn get-article-with-whitespace
  [token]
  (if (maybe/just? (:article token))
    (str ((cond (or (:article-title token) (:start token)) str/capitalize
                (:upper token) str/upper-case
                :else identity)
           @(:article token)) " ")
    ""))

(defn inflect-case
  [token]
  ((cond (and (:start token)
              (maybe/just? (:article token))
              ;TODO check for proper adjectives
              (not (contains? proper-nouns (:tag_ token))))
         str/lower-case
         (:is_title token) str/capitalize
         (:upper token) str/upper-case
         :else identity)
    (if (:inflected token)
      (str (get (set/map-invert word)
                (:lower_ token)
                (if (#{"VBP" "VBZ"} (:tag_ token))
                  (:verb token)
                  (:noun token)))
           (:whitespace_ token))
      (:text_with_ws token))))

(defn generate
  [coll]
  (binding [*read-eval* false]
    (read-string (:body (client/post "http://localhost:3000"
                                     {:content-type :application/edn
                                      :body         (pr-str coll)})))))


(defn get-native
  [text]
  (str/join (map (fn [token]
                   (str (get-article-with-whitespace token)
                        (inflect-case token)))
                 (map merge
                      (merge-inferred (infer (map :lower_ (supplement-parsed (parse text))))
                                      (supplement-parsed (parse text)))
                      (generate (map :lower_ (supplement-parsed (parse text))))))))

(def cors-headers
  {"Access-Control-Allow-Origin"  "*"
   "Access-Control-Allow-Headers" "Content-Type"
   "Access-Control-Allow-Methods" "GET,POST,OPTIONS"})

(defn app
  [request]
  (if (:text (:params request))
    {:status  200
     :body    (pr-str {:text (get-native (:text (:params request)))})
     :headers cors-headers}
    {:status  200
     :body    "hello world"
     :headers cors-headers}))

(def start
  (partial web/run
           (-> app
               wrap-edn-params
               #_(wrap-cors :access-control-allow-origin #".*"
                            :access-control-allow-methods [:get :put :post :delete]))
           {:host "0.0.0.0"}))

(def stop
  (partial web/stop))

(defstate server
          :start (start)
          :stop (stop))
